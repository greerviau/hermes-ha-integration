"""Conversation agent for Hermes Agent."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any

from homeassistant.components.conversation import (
    MATCH_ALL,
    AbstractConversationAgent,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
    async_set_agent,
    async_unset_agent,
)
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent, template

from .api import HermesApiClient, HermesApiError, HermesStreamSetupError
from .compat import entry_value, resolve_continued_conversation_mode
from .const import (
    CONF_ALWAYS_SPEAK_FALLBACK,
    CONF_API_KEY,
    CONF_CONTEXT_MAX_CHARS,
    CONF_ENABLE_SESSION_REUSE,
    CONF_EXPOSE_DEVICE_CONTEXT,
    CONF_FALLBACK_MEDIA_PLAYER,
    CONF_FALLBACK_TTS_ENGINE,
    CONF_INCLUDE_EXPOSED_ENTITIES,
    CONF_PROMPT,
    CONF_SESSION_TIMEOUT_SECONDS,
    CONF_SPEECH_NORMALIZATION,
    DEFAULT_ALWAYS_SPEAK_FALLBACK,
    DEFAULT_CONTEXT_MAX_CHARS,
    DEFAULT_ENABLE_SESSION_REUSE,
    DEFAULT_EXPOSE_DEVICE_CONTEXT,
    DEFAULT_FALLBACK_MEDIA_PLAYER,
    DEFAULT_FALLBACK_TTS_ENGINE,
    DEFAULT_INCLUDE_EXPOSED_ENTITIES,
    DEFAULT_MAX_HISTORY_MESSAGES,
    DEFAULT_PROMPT,
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    DEFAULT_SPEECH_NORMALIZATION,
    DOMAIN,
    FOLLOW_UP_MODE_ALWAYS,
    FOLLOW_UP_MODE_AUTO,
    LEGACY_CONF_INSTRUCTIONS,
)

try:
    from homeassistant.components.conversation import ChatLog, async_get_chat_log
    from homeassistant.helpers.chat_session import async_get_chat_session
except ImportError:
    ChatLog = Any
    async_get_chat_log = None
    async_get_chat_session = None

_LOGGER = logging.getLogger(__name__)
_MAX_CACHED_CONVERSATIONS = 50
_QUESTION_MARKERS = ("?", "\uFF1F")
_TRAILING_CLOSERS = "\"')]}" + "\u201d\u2019\u00bb"
_AUTO_FOLLOW_UP_PROMPT = (
    "When voice auto follow-up is active and you want the user to reply, "
    "give any needed answer first and end with one short, direct question as "
    "the final sentence. Do not add any words after the question mark."
)

_UNSAFE_SPEECH_TAG_PATTERN = (
    "think|analysis|tool_call|tool_calls|function_call|function_calls|"
    "tool_result|tool_results"
)
_UNSAFE_SPEECH_BLOCK_RE = re.compile(
    rf"<\s*({_UNSAFE_SPEECH_TAG_PATTERN})\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_UNSAFE_SPEECH_OPEN_RE = re.compile(
    rf"<\s*(?:{_UNSAFE_SPEECH_TAG_PATTERN})\b[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_UNSAFE_SPEECH_TAG_RE = re.compile(
    rf"<\s*/?\s*(?:{_UNSAFE_SPEECH_TAG_PATTERN})\b[^>]*>",
    re.IGNORECASE,
)
_UNSAFE_SPEECH_START_RE = re.compile(
    rf"<\s*({_UNSAFE_SPEECH_TAG_PATTERN})\b[^>]*>",
    re.IGNORECASE,
)
_MAX_UNSAFE_CLOSE_TAG_LENGTH = 80
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: Callable[[list[Any]], None],
) -> None:
    """Set up the Hermes conversation entity."""
    entry_data = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            HermesConversationAgent(
                hass,
                entry,
                entry_data["client"],
                session_map=entry_data["sessions"],
            )
        ]
    )


class _UnsafeSpeechStreamFilter:
    """Incrementally drop hidden reasoning and tool markup from streamed speech."""

    def __init__(self, *, normalize_speech: bool = False) -> None:
        self._buffer = ""
        self._discard_until_tag: str | None = None
        self._normalize_speech = normalize_speech

    def feed(self, text: str) -> str:
        """Add a stream delta and return the safe text that can be emitted now."""
        if not text:
            return ""
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> str:
        """Return any remaining safe text at end of stream."""
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        safe_parts: list[str] = []

        while self._buffer:
            if self._discard_until_tag:
                close_re = re.compile(
                    rf"<\s*/\s*{re.escape(self._discard_until_tag)}\s*>",
                    re.IGNORECASE,
                )
                close_match = close_re.search(self._buffer)
                if close_match is None:
                    if final:
                        self._buffer = ""
                        self._discard_until_tag = None
                    else:
                        self._buffer = self._buffer[-_MAX_UNSAFE_CLOSE_TAG_LENGTH:]
                    break

                self._buffer = self._buffer[close_match.end() :]
                self._discard_until_tag = None
                continue

            open_match = _UNSAFE_SPEECH_START_RE.search(self._buffer)
            if open_match is not None:
                safe_parts.append(self._buffer[: open_match.start()])
                tag = open_match.group(1).lower()
                close_re = re.compile(
                    rf"<\s*/\s*{re.escape(tag)}\s*>",
                    re.IGNORECASE,
                )
                close_match = close_re.search(self._buffer, open_match.end())
                if close_match is None:
                    self._buffer = self._buffer[open_match.end() :]
                    self._discard_until_tag = tag
                    if not final:
                        self._buffer = self._buffer[-_MAX_UNSAFE_CLOSE_TAG_LENGTH:]
                    else:
                        self._buffer = ""
                        self._discard_until_tag = None
                    break

                self._buffer = self._buffer[close_match.end() :]
                continue

            safe_parts.append(self._consume_safe_buffer(final=final))
            break

        return _sanitize_stream_text_for_speech(
            "".join(safe_parts), normalize_speech=self._normalize_speech
        )

    def _consume_safe_buffer(self, *, final: bool) -> str:
        if final:
            safe = self._buffer
            self._buffer = ""
            return safe

        last_lt = self._buffer.rfind("<")
        if last_lt != -1 and ">" not in self._buffer[last_lt:]:
            safe = self._buffer[:last_lt]
            self._buffer = self._buffer[last_lt:]
            return safe

        safe = self._buffer
        self._buffer = ""
        return safe


def _remove_unsafe_speech_markup(text: str) -> str:
    """Remove hidden reasoning and tool-call markup before it reaches TTS."""
    cleaned = text
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _UNSAFE_SPEECH_BLOCK_RE.sub("", cleaned)
    cleaned = _UNSAFE_SPEECH_OPEN_RE.sub("", cleaned)
    return _UNSAFE_SPEECH_TAG_RE.sub("", cleaned)


def _sanitize_stream_text_for_speech(
    text: str, *, normalize_speech: bool = False
) -> str:
    """Apply safe, local cleanup to a speech stream delta."""
    if not text:
        return text
    cleaned = text.replace("\r\n", "\n")
    cleaned = _remove_unsafe_speech_markup(cleaned)
    cleaned = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = (
        cleaned.replace("```", "")
        .replace("`", "")
        .replace("**", "")
        .replace("__", "")
        .replace("~~", "")
    )
    return _normalize_speech_text(cleaned) if normalize_speech else cleaned


def _normalize_speech_text(text: str) -> str:
    """Expand conservative, common English forms for speech."""
    if not text:
        return text

    # Preserve leading/trailing whitespace because this helper is also used on
    # individual streaming chunks. Stripping each chunk joins adjacent words.
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    text = text.strip()
    if not text:
        return leading + trailing

    def replace_currency(match: re.Match[str]) -> str:
        amount = match.group(1)
        if "." in amount:
            dollars, cents = amount.split(".", 1)
            cents = cents.ljust(2, "0")
            return (
                f"{_number_to_words(int(dollars))} dollars and "
                f"{_number_to_words(int(cents))} cents"
            )
        return f"{_number_to_words(int(amount))} dollars"

    normalized = re.sub(r"\$(\d+(?:\.\d{1,2})?)", replace_currency, text)
    normalized = re.sub(r"(?<=\d)%", " percent", normalized)
    normalized = normalized.replace("≈", "approximately")
    normalized = normalized.replace("&", " and ")

    def replace_date(match: re.Match[str]) -> str:
        year, month, day = (int(part) for part in match.groups())
        try:
            date(year, month, day)
        except ValueError:
            return match.group(0)
        return f"{_MONTH_NAMES[month - 1]} {_ordinal_to_words(day)}, {_year_to_words(year)}"

    # Handle dates before protecting subtraction expressions (the date hyphens
    # must not be mistaken for minus signs).
    normalized = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", replace_date, normalized)

    # Do not alter identifiers, URLs, IPs, UUIDs, versions, or arithmetic.
    protected: list[str] = []

    def protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"__HERMES_PROTECTED_{len(protected) - 1}__"

    normalized = re.sub(
        r"(?:https?://|www\.)\S+|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[A-Za-z_][\w.-]*\d[\w.-]*\b|(?<!\w)\d+(?:\s*[+*/-]\s*\d+)+(?!\w)",
        protect,
        normalized,
        flags=re.IGNORECASE,
    )

    def replace_time(match: re.Match[str]) -> str:
        hour, minute = (int(part) for part in match.groups())
        if hour > 23 or minute > 59:
            return match.group(0)
        if hour == 0 and minute == 0:
            return "midnight"
        if hour == 12 and minute == 0:
            return "noon"
        suffix = "AM" if hour < 12 else "PM"
        spoken_hour = _number_to_words(hour % 12 or 12)
        if minute == 0:
            spoken_minute = ""
        elif minute < 10:
            spoken_minute = f"oh {_number_to_words(minute)}"
        else:
            spoken_minute = _number_to_words(minute)
        return f"{spoken_hour}{f' {spoken_minute}' if spoken_minute else ''} {suffix}"

    normalized = re.sub(r"\b(\d{1,2}):(\d{2})\b", replace_time, normalized)
    normalized = re.sub(
        r"(?<![\w.])(\d{1,3}(?:,\d{3})*|\d+)(?![\w.])",
        lambda match: _number_to_words(int(match.group(1).replace(",", ""))),
        normalized,
    )
    for index, original in enumerate(protected):
        normalized = normalized.replace(f"__HERMES_PROTECTED_{index}__", original)
    normalized = re.sub(r"\s+", " ", normalized)
    return leading + normalized.strip() + trailing


def _ordinal_to_words(number: int) -> str:
    """Convert the day of a month to a short English ordinal."""
    irregular = {
        1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth",
        9: "ninth", 12: "twelfth",
    }
    if number in irregular:
        return irregular[number]
    if 10 < number < 14:
        return f"{_number_to_words(number)}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{_number_to_words(number)}{suffix}"


def _year_to_words(year: int) -> str:
    """Speak a four-digit year in its usual conversational form."""
    if 1000 <= year <= 2099:
        return f"{_number_to_words(year // 100)} {_number_to_words(year % 100)}" if year % 100 else _number_to_words(year // 100) + " hundred"
    return _number_to_words(year)


def _number_to_words(number: int) -> str:
    """Convert a non-negative integer into concise English words."""
    ones = (
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    if number < 20:
        return ones[number]
    if number < 100:
        return tens[number // 10] + (f"-{ones[number % 10]}" if number % 10 else "")
    if number < 1000:
        remainder = number % 100
        return f"{ones[number // 100]} hundred" + (f" {_number_to_words(remainder)}" if remainder else "")
    for scale, name in ((1_000_000, "million"), (1_000, "thousand")):
        if number >= scale:
            remainder = number % scale
            return f"{_number_to_words(number // scale)} {name}" + (f" {_number_to_words(remainder)}" if remainder else "")
    return str(number)


def _sanitize_text_for_speech(
    text: str, *, normalize_speech: bool = False
) -> str:
    """Convert markdown-ish assistant output into plain speech-friendly text."""
    if not text:
        return text

    cleaned = text.replace("\r\n", "\n")
    cleaned = _remove_unsafe_speech_markup(cleaned)
    cleaned = re.sub(r"```(?:[\w+-]+)?\n?(.*?)```", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^>+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
    cleaned = re.sub(r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!_)_(?!\s)(.*?)(?<!\s)_(?!_)", r"\1", cleaned)
    cleaned = re.sub(r"~~(.*?)~~", r"\1", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\[[^\]]*\]", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = cleaned.strip()
    return _normalize_speech_text(cleaned) if normalize_speech else cleaned


class HermesConversationAgent(ConversationEntity, AbstractConversationAgent):
    """Hermes Agent conversation entity for Home Assistant."""

    _attr_should_poll = False
    _attr_supports_streaming = True

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        client: HermesApiClient,
        session_map: dict[str, dict[str, Any]],
    ) -> None:
        """Initialise the conversation agent."""
        self.hass = hass
        self.entry = entry
        self.client = client
        self.session_map = session_map
        self._attr_unique_id = entry.entry_id
        self._attr_name = getattr(entry, "title", None) or "Hermes Agent"
        self._attr_supported_features = ConversationEntityFeature.CONTROL
        # conversation_id -> list of {"role": ..., "content": ...}
        self._history: OrderedDict[str, list[dict[str, str]]] = OrderedDict()

    @property
    def supported_languages(self) -> list[str] | str:
        """Return supported languages (all — the LLM handles it)."""
        return MATCH_ALL

    @property
    def supports_streaming(self) -> bool:
        """Return if the entity supports streaming responses."""
        return True

    async def async_added_to_hass(self) -> None:
        """Register a legacy agent alias for older Home Assistant callers."""
        if super_added := getattr(super(), "async_added_to_hass", None):
            await super_added()
        async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Remove the legacy agent alias when Home Assistant unloads the entity."""
        async_unset_agent(self.hass, self.entry)
        if super_removed := getattr(super(), "async_will_remove_from_hass", None):
            await super_removed()

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Handle a modern Home Assistant conversation turn."""
        return await self._async_process_with_error_handling(user_input, chat_log)

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a conversation turn."""
        return await self._async_process_with_error_handling(user_input)

    async def _async_process_with_error_handling(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog | None = None,
    ) -> ConversationResult:
        """Process a conversation turn and convert unexpected errors."""
        try:
            if (
                chat_log is None
                and async_get_chat_log is not None
                and async_get_chat_session is not None
            ):
                with (
                    async_get_chat_session(
                        self.hass,
                        user_input.conversation_id,
                    ) as session,
                    async_get_chat_log(
                        self.hass,
                        session,
                        user_input,
                    ) as active_chat_log,
                ):
                    return await self._async_process_inner(
                        user_input, chat_log=active_chat_log
                    )
            return await self._async_process_inner(user_input, chat_log=chat_log)
        except Exception:
            _LOGGER.exception("Unexpected error in async_process")
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "An internal error occurred. Check the logs.",
            )
            return self._build_conversation_result(
                intent_response,
                getattr(chat_log, "conversation_id", None)
                or user_input.conversation_id
                or "default",
                continue_conversation=False,
            )

    async def _async_process_inner(
        self, user_input: ConversationInput, chat_log: ChatLog | None = None
    ) -> ConversationResult:
        """Inner processing — wrapped by async_process for error logging."""
        conv_id = (
            getattr(chat_log, "conversation_id", None)
            or user_input.conversation_id
            or str(uuid.uuid4())
        )
        follow_up_mode = self._continued_conversation_mode()
        session_reuse = self._session_reuse_enabled()
        session_key = self._build_session_key(user_input, conv_id) if session_reuse else None
        session_id = self._get_active_session_id(session_key) if session_key else None

        # Resolve username from HA auth
        user_name = await self._get_user_name(user_input)

        # Build system prompt (optional — Hermes Agent has its own)
        system_prompt = self._render_system_prompt(user_name, user_input)

        # Append extra system prompt from HA voice pipeline if present
        extra = getattr(user_input, "extra_system_prompt", None)
        if extra:
            system_prompt = (system_prompt + "\n\n" + extra) if system_prompt else extra

        # Append origin context when requested
        if self._device_context_enabled():
            context_lines = self._build_origin_context(user_input)
            if context_lines:
                origin_block = "Origin context:\n" + "\n".join(f"- {line}" for line in context_lines)
                system_prompt = (system_prompt + "\n\n" + origin_block) if system_prompt else origin_block

        system_prompt = self._append_auto_follow_up_prompt(
            system_prompt,
            follow_up_mode,
        )

        if session_reuse:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_input.text})
        else:
            history = self._history.setdefault(conv_id, [])
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history)
            messages.append({"role": "user", "content": user_input.text})

        try:
            if chat_log is None:
                response_text = await self._get_response(messages, session_id=session_id)
            else:
                response_text = await self._stream_chat_log_response(
                    chat_log,
                    messages,
                    session_id=session_id,
                )
            display_text = _sanitize_text_for_speech(response_text)
            spoken_text = _sanitize_text_for_speech(
                display_text,
                normalize_speech=self._speech_normalization_enabled(),
            )
        except HermesApiError as err:
            _LOGGER.error("Hermes API error: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Error communicating with Hermes Agent: {err}",
            )
            return self._build_conversation_result(
                intent_response,
                conv_id,
                continue_conversation=False,
            )

        if session_key:
            self._remember_session(session_key, self.client.last_session_id)

        if not session_reuse:
            history = self._history.setdefault(conv_id, [])
            history.append({"role": "user", "content": user_input.text})
            history.append({"role": "assistant", "content": display_text})
            self._history.move_to_end(conv_id)

            while len(history) > DEFAULT_MAX_HISTORY_MESSAGES:
                history.pop(0)
                if history and history[0]["role"] == "assistant":
                    history.pop(0)

            while len(self._history) > _MAX_CACHED_CONVERSATIONS:
                self._history.popitem(last=False)

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(spoken_text)
        await self._async_speak_fallback(spoken_text, user_input)
        continue_conversation = self._should_continue_conversation(
            follow_up_mode,
            display_text,
        )

        return self._build_conversation_result(
            intent_response,
            conv_id,
            continue_conversation=continue_conversation,
        )

    async def _stream_chat_log_response(
        self,
        chat_log: ChatLog,
        messages: list[dict[str, str]],
        session_id: str | None = None,
    ) -> str:
        """Stream safe assistant deltas into Home Assistant's chat log."""
        chunks: list[str] = []

        async def _stream() -> AsyncIterator[dict[str, str]]:
            started = False
            try:
                async for chunk in self._iter_voice_safe_response(
                    messages, session_id=session_id
                ):
                    if not chunk:
                        continue
                    if not started:
                        yield {"role": "assistant"}
                        started = True
                    chunks.append(chunk)
                    yield {"content": chunk}
            except HermesApiError as err:
                if chunks:
                    _LOGGER.warning(
                        "Hermes stream failed after content started; keeping partial response: %s",
                        err,
                    )
                    return
                raise

        try:
            async for _content in chat_log.async_add_delta_content_stream(
                self._agent_id(), _stream()
            ):
                pass
        except HermesApiError as err:
            if chunks:
                _LOGGER.warning(
                    "Hermes stream failed after content started; keeping partial response: %s",
                    err,
                )
                return "".join(chunks)
            raise

        return "".join(chunks)

    async def _iter_voice_safe_response(
        self,
        messages: list[dict[str, str]],
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield speech-safe assistant text chunks from Hermes streaming."""
        # Normalize only after the complete response is assembled. Applying
        # it to individual stream chunks strips their boundary whitespace.
        speech_filter = _UnsafeSpeechStreamFilter()
        stream_started = False

        try:
            async for chunk in self.client.async_stream_message(
                messages, session_id=session_id
            ):
                stream_started = True
                if safe_chunk := speech_filter.feed(chunk):
                    yield safe_chunk
        except HermesStreamSetupError as err:
            if stream_started:
                raise
            _LOGGER.debug(
                "Hermes streaming setup failed; falling back to non-streaming: %s",
                err,
            )
            result = await self.client.async_send_message(messages, session_id=session_id)
            if safe_text := _sanitize_text_for_speech(result.text):
                yield safe_text
            return

        if final_chunk := speech_filter.flush():
            yield final_chunk

    def _agent_id(self) -> str:
        """Return the best available Home Assistant agent identifier."""
        return getattr(self, "entity_id", None) or self.entry.entry_id

    def _speech_normalization_enabled(self) -> bool:
        """Return whether optional speech symbol normalization is enabled."""
        return bool(
            entry_value(
                self.entry,
                CONF_SPEECH_NORMALIZATION,
                DEFAULT_SPEECH_NORMALIZATION,
            )
        )

    async def _get_response(
        self,
        messages: list[dict[str, str]],
        session_id: str | None = None,
    ) -> str:
        """Get a response from the API using voice-safe streaming."""
        chunks: list[str] = []
        try:
            async for chunk in self._iter_voice_safe_response(messages, session_id):
                chunks.append(chunk)
        except HermesStreamSetupError as err:
            _LOGGER.debug(
                "Hermes streaming setup failed; falling back to non-streaming: %s",
                err,
            )
            result = await self.client.async_send_message(messages, session_id=session_id)
            return result.text
        except HermesApiError:
            if chunks:
                _LOGGER.warning(
                    "Hermes stream failed after content started; not retrying to avoid duplicate tool calls"
                )
            raise
        return "".join(chunks)

    async def _get_user_name(self, user_input: ConversationInput) -> str:
        """Resolve the display name of the user from HA auth."""
        try:
            context = getattr(user_input, "context", None)
            if context is None:
                return "the user"
            user_id = getattr(context, "user_id", None)
            if not user_id:
                return "the user"
            user = await self.hass.auth.async_get_user(user_id)
            if user and user.name:
                return user.name
        except Exception:
            _LOGGER.debug("Could not resolve username", exc_info=True)
        return "the user"

    def _render_system_prompt(
        self,
        user_name: str,
        user_input: ConversationInput | None = None,
    ) -> str:
        """Render the system prompt template with HA context."""
        prompt_template = entry_value(
            self.entry,
            CONF_PROMPT,
            DEFAULT_PROMPT,
            legacy_keys=(LEGACY_CONF_INSTRUCTIONS,),
        )
        if not prompt_template:
            return ""

        variables: dict[str, Any] = {
            "ha_name": self.hass.config.location_name,
            "user_name": user_name,
            **self._get_origin_prompt_variables(user_input),
        }

        include_entities = entry_value(
            self.entry,
            CONF_INCLUDE_EXPOSED_ENTITIES,
            DEFAULT_INCLUDE_EXPOSED_ENTITIES,
        )
        if include_entities:
            variables["exposed_entities"] = self._get_exposed_entities()
        else:
            variables["exposed_entities"] = []

        try:
            tpl = template.Template(prompt_template, self.hass)
            return tpl.async_render(variables)
        except template.TemplateError as err:
            _LOGGER.warning("System prompt template error: %s", err)
            return prompt_template

    def _append_auto_follow_up_prompt(
        self,
        system_prompt: str,
        follow_up_mode: str,
    ) -> str:
        """Append guidance that makes auto follow-up turns cleaner."""
        if follow_up_mode != FOLLOW_UP_MODE_AUTO:
            return system_prompt

        if system_prompt:
            return f"{system_prompt}\n\n{_AUTO_FOLLOW_UP_PROMPT}"

        return _AUTO_FOLLOW_UP_PROMPT

    def _get_origin_prompt_variables(
        self, user_input: ConversationInput | None
    ) -> dict[str, str]:
        """Return exact origin IDs and a paired media player, when available."""
        if user_input is None:
            return {
                "origin_satellite": "",
                "origin_media_player": "",
                "origin_device": "",
            }

        satellite_id = getattr(user_input, "satellite_id", None) or ""
        device_id = getattr(user_input, "device_id", None) or ""
        entity_reg = er.async_get(self.hass)

        # A satellite is normally an entity whose registry entry points to its device.
        if not device_id and satellite_id:
            try:
                satellite_entry = entity_reg.async_get(satellite_id)
            except (AttributeError, TypeError, ValueError):
                satellite_entry = None
            device_id = getattr(satellite_entry, "device_id", None) or ""

        media_player = ""
        if device_id:
            entries: Any = ()
            try:
                entries_for_device = getattr(entity_reg, "async_entries_for_device", None)
                if callable(entries_for_device):
                    entries = entries_for_device(device_id)
                else:
                    entries = entity_reg.async_entries()
            except (AttributeError, TypeError, ValueError):
                entries = ()
            for entity_entry in entries:
                entity_id = getattr(entity_entry, "entity_id", "") or ""
                domain = (
                    getattr(entity_entry, "domain", None)
                    or entity_id.partition(".")[0]
                )
                if (
                    domain == "media_player"
                    and getattr(entity_entry, "device_id", None) == device_id
                ):
                    media_player = entity_id
                    break

        return {
            "origin_satellite": satellite_id,
            "origin_media_player": media_player,
            "origin_device": device_id,
        }

    def _get_exposed_entities(self) -> list[dict[str, Any]]:
        """Get a list of entities exposed to the conversation agent."""
        max_chars = entry_value(
            self.entry,
            CONF_CONTEXT_MAX_CHARS,
            DEFAULT_CONTEXT_MAX_CHARS,
        )
        entity_reg = er.async_get(self.hass)
        device_reg = dr.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        get_entity_aliases = getattr(er, "async_get_entity_aliases", None)
        entities: list[dict[str, Any]] = []
        total_chars = 0

        for state in self.hass.states.async_all():
            try:
                if not async_should_expose(
                    self.hass, "conversation", state.entity_id
                ):
                    continue
            except Exception:
                continue

            registry_entry = entity_reg.async_get(state.entity_id)
            raw_aliases = (
                get_entity_aliases(self.hass, registry_entry)
                if registry_entry is not None and get_entity_aliases is not None
                else (getattr(registry_entry, "aliases", None) or ())
            )
            aliases = sorted(
                alias.strip()
                for alias in raw_aliases
                if isinstance(alias, str) and alias.strip()
            )
            area_id = getattr(registry_entry, "area_id", None)
            device_id = getattr(registry_entry, "device_id", None)
            if not area_id and device_id:
                device = device_reg.async_get(device_id)
                area_id = getattr(device, "area_id", None)
            area = area_reg.async_get_area(area_id) if area_id else None

            entity_info: dict[str, Any] = {
                "entity_id": state.entity_id,
                "name": state.attributes.get("friendly_name", state.entity_id),
                "state": str(state.state),
                "domain": state.entity_id.partition(".")[0],
                "aliases": aliases,
                "area": getattr(area, "name", "") if area else "",
            }

            details = [
                f"state={entity_info['state']}",
                f"domain={entity_info['domain']}",
            ]
            if aliases:
                details.append(f"aliases={' / '.join(aliases)}")
            if entity_info["area"]:
                details.append(f"area={entity_info['area']}")
            line = (
                f"- {entity_info['entity_id']} ({entity_info['name']}): "
                + ", ".join(details)
            )
            total_chars += len(line) + 1
            if total_chars > max_chars:
                break
            entities.append(entity_info)

        return entities

    def _continued_conversation_mode(self) -> str:
        return resolve_continued_conversation_mode(self.entry)

    def _should_continue_conversation(
        self,
        follow_up_mode: str,
        response_text: str,
    ) -> bool:
        """Return whether the voice pipeline should keep listening."""
        if follow_up_mode == FOLLOW_UP_MODE_ALWAYS:
            return True
        if follow_up_mode != FOLLOW_UP_MODE_AUTO:
            return False
        return self._response_invites_follow_up(response_text)

    def _response_invites_follow_up(self, response_text: str) -> bool:
        """Detect when Hermes ended with a question worth keeping HA open for."""
        stripped_text = response_text.strip().rstrip(_TRAILING_CLOSERS)
        return stripped_text.endswith(_QUESTION_MARKERS)

    def _session_reuse_enabled(self) -> bool:
        if not bool(
            entry_value(
                self.entry,
                CONF_ENABLE_SESSION_REUSE,
                DEFAULT_ENABLE_SESSION_REUSE,
            )
        ):
            return False

        # Hermes Agent deliberately rejects X-Hermes-Session-Id continuation
        # unless the API server is protected by API-key authentication.  If the
        # user configured this integration without an API key, do not send the
        # session header; otherwise the second voice turn would fail with 403.
        api_key = entry_value(
            self.entry,
            CONF_API_KEY,
            "",
            prefer_options=False,
        )
        if not api_key:
            _LOGGER.debug("Hermes session reuse disabled because no API key is configured")
            return False

        return True

    def _session_timeout_seconds(self) -> int:
        try:
            return max(
                0,
                int(
                    entry_value(
                        self.entry,
                        CONF_SESSION_TIMEOUT_SECONDS,
                        DEFAULT_SESSION_TIMEOUT_SECONDS,
                    )
                ),
            )
        except (TypeError, ValueError):
            return DEFAULT_SESSION_TIMEOUT_SECONDS

    def _device_context_enabled(self) -> bool:
        return bool(
            entry_value(
                self.entry,
                CONF_EXPOSE_DEVICE_CONTEXT,
                DEFAULT_EXPOSE_DEVICE_CONTEXT,
            )
        )

    def _build_session_key(self, user_input: ConversationInput, conversation_id: str) -> str:
        device_id = getattr(user_input, "device_id", None)
        satellite_id = getattr(user_input, "satellite_id", None)
        if device_id:
            return f"device:{device_id}"
        if satellite_id:
            return f"satellite:{satellite_id}"
        return f"conversation:{conversation_id}"

    def _get_active_session_id(self, session_key: str | None) -> str | None:
        if not session_key:
            return None

        record = self.session_map.get(session_key)
        if not record:
            return None

        session_id = record.get("session_id")
        last_used_at = float(record.get("last_used_at", 0) or 0)
        timeout_seconds = self._session_timeout_seconds()
        if timeout_seconds and (time.time() - last_used_at) > timeout_seconds:
            self.session_map.pop(session_key, None)
            return None

        if isinstance(session_id, str) and session_id.strip():
            return session_id
        return None

    def _remember_session(self, session_key: str, session_id: str | None) -> None:
        if not session_id:
            self.session_map.pop(session_key, None)
            return

        self.session_map[session_key] = {
            "session_id": session_id,
            "last_used_at": time.time(),
        }

    def _build_origin_context(self, user_input: ConversationInput) -> list[str]:
        lines: list[str] = []
        language = getattr(user_input, "language", None)
        device_id = getattr(user_input, "device_id", None)
        satellite_id = getattr(user_input, "satellite_id", None)

        if language:
            lines.append(f"Language: {language}")
        if device_id:
            lines.extend(self._describe_device(device_id))
            lines.append(f"Origin device_id: {device_id}")
        if satellite_id:
            lines.extend(self._describe_satellite(satellite_id))
            lines.append(f"Origin satellite_id: {satellite_id}")
        media_player = self._get_origin_prompt_variables(user_input)[
            "origin_media_player"
        ]
        if media_player:
            lines.append(f"Origin media_player: {media_player}")
        return lines

    def _describe_device(self, device_id: str) -> list[str]:
        device_reg = dr.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        device = device_reg.async_get(device_id)
        if not device:
            return [f"Home Assistant device_id: {device_id}"]

        lines = [f"Origin device: {device.name_by_user or device.name or device_id}"]
        if device.area_id:
            area = area_reg.async_get_area(device.area_id)
            if area:
                lines.append(f"Origin area: {area.name}")
        return lines

    def _describe_satellite(self, satellite_id: str) -> list[str]:
        state = self.hass.states.get(satellite_id) if "." in satellite_id else None
        if not state:
            return [f"Assist satellite: {satellite_id}"]
        friendly_name = state.attributes.get("friendly_name", satellite_id)
        return [f"Assist satellite: {friendly_name} ({satellite_id})"]

    def _build_conversation_result(
        self,
        intent_response: intent.IntentResponse,
        conversation_id: str,
        *,
        continue_conversation: bool = False,
    ) -> ConversationResult:
        """Build a conversation result, preserving compatibility with older HA."""
        try:
            return ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
                continue_conversation=continue_conversation,
            )
        except TypeError:
            return ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

    async def _async_speak_fallback(
        self, text: str, user_input: ConversationInput
    ) -> None:
        if not text.strip():
            return

        if not (
            getattr(user_input, "device_id", None)
            or getattr(user_input, "satellite_id", None)
        ):
            return

        speak_fallback = entry_value(
            self.entry,
            CONF_ALWAYS_SPEAK_FALLBACK,
            DEFAULT_ALWAYS_SPEAK_FALLBACK,
        )
        if not speak_fallback:
            return

        media_player_entity = entry_value(
            self.entry,
            CONF_FALLBACK_MEDIA_PLAYER,
            DEFAULT_FALLBACK_MEDIA_PLAYER,
        )
        tts_entity = entry_value(
            self.entry,
            CONF_FALLBACK_TTS_ENGINE,
            DEFAULT_FALLBACK_TTS_ENGINE,
        )
        if not media_player_entity or not tts_entity:
            return

        service_data = {
            "entity_id": tts_entity,
            "media_player_entity_id": media_player_entity,
            "message": text,
            "cache": True,
        }

        language = getattr(user_input, "language", None)
        if language:
            service_data["language"] = language

        try:
            await self.hass.services.async_call(
                "tts",
                "speak",
                service_data,
                blocking=True,
            )
        except Exception as err:
            _LOGGER.warning("Fallback TTS failed: %s", err)
