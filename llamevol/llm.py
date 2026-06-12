"""LLM manager to connect to different types of models via LangChain."""

import os
import logging
from typing import Optional, Any
from collections.abc import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
ONEHUB_API_KEY = os.getenv("ONEHUB_API_KEY")
ONEHUB_BASE_URL = os.getenv("ONEHUB_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LLMS = {
    "gemini-2.0-flash-exp": (
        "gemini-2.0-flash-exp",
        GEMINI_API_KEY,
        GEMINI_BASE_URL,
        10,
        "google",
    ),
    "gemini-2.0-flash": (
        "gemini-2.0-flash",
        GEMINI_API_KEY,
        GEMINI_BASE_URL,
        10,
        "google",
    ),
    "gemini-2.5-flash": (
        "gemini-2.5-flash",
        GEMINI_API_KEY,
        GEMINI_BASE_URL,
        10,
        "google",
    ),
    "onehub-gemini-2.0-flash": (
        "gemini-2.0-flash-exp",
        ONEHUB_API_KEY,
        ONEHUB_BASE_URL,
        5,
        "openrouter",
    ),
    "onehub-gemma2-9b-it": (
        "gemma2-9b-it",
        ONEHUB_API_KEY,
        ONEHUB_BASE_URL,
        5,
        "openrouter",
    ),
    "o_deepseek-r1-free": (
        "deepseek/deepseek-r1-0528:free",
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        5,
        "openrouter",
    ),
    "o_deepseek-r1": (
        "deepseek/deepseek-r1-0528",
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        5,
        "openrouter",
    ),
    "o_qwen3-coder-free": (
        "qwen/qwen3-coder:free",
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        5,
        "openrouter",
    ),
    "o_qwen3-coder": (
        "qwen/qwen3-coder",
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        5,
        "openrouter",
    ),
    "o_gpt4o": (
        "openai/gpt-4o-2024-11-20",
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        5,
        "openrouter",
    ),
    "o_gemini-2.0-flash": (
        "google/gemini-2.0-flash-001",
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        5,
        "openrouter",
    ),
    "gpt-4o": ("gpt-4o-2024-11-20", OPENAI_API_KEY, None, 5, "openai"),
}

_SUPPORTED_KWARGS = {
    "google": {"temperature", "top_k", "top_p"},
    "vertex": {"temperature", "top_k", "top_p"},
    "openai": {"temperature", "top_p"},
    "openrouter": {"temperature", "top_p"},
    "request": {"temperature", "top_p"},
    "anthropic": {"temperature", "top_p", "max_tokens"},
}


class LLMClientResponse:
    def __init__(self, response):
        self.response = response
        self.text = None
        self._prompt_token_count = 0
        self._response_token_count = 0
        self.error = None

    @property
    def prompt_token_count(self):
        return self._prompt_token_count

    @prompt_token_count.setter
    def prompt_token_count(self, value):
        if value is not None:
            self._prompt_token_count = value

    @property
    def response_token_count(self):
        return self._response_token_count

    @response_token_count.setter
    def response_token_count(self, value):
        if value is not None:
            self._response_token_count = value

    def __str__(self):
        if self.error is not None:
            return str(self.error)
        return str(self.response)


def _create_model(
    client_str, model_name, api_key, base_url, project=None, location=None
):
    if client_str == "google":
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    if client_str == "vertex":
        # project/location are configurable (default to the team project/region).
        return ChatVertexAI(
            model_name=model_name,
            project=project or "starry-seat-441021-m2",
            location=location or "us-central1",
        )

    if client_str in ("openai", "openrouter", "request"):
        kwargs = {"model": model_name, "api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    if client_str == "anthropic":
        # Pass api_key when provided; otherwise ChatAnthropic reads ANTHROPIC_API_KEY.
        kwargs = {"model": model_name, "max_tokens": 16384}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    raise ValueError(f"Unsupported client_str: {client_str}")


def _convert_messages(session_messages):
    lc_messages = []
    for msg in session_messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


class LLMmanager:
    def __init__(
        self,
        model_key: str = None,
        model_name: str = None,
        api_key: str = None,
        base_url: str = None,
        client_str: str = None,
        project: str = None,
        location: str = None,
    ):
        use_vertex = (client_str == "vertex") or (
            os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "1"
        )

        if model_key is None:
            if model_name is None:
                raise ValueError("model_name must be provided.")

            keyless_providers = {"anthropic"}
            if not (api_key or use_vertex or client_str in keyless_providers):
                raise ValueError(
                    "Provide api_key (Gemini Developer API) or enable Vertex "
                    "(set client_str='vertex' or GOOGLE_GENAI_USE_VERTEXAI=1)."
                )

            _model = (model_name, api_key, base_url, 5, client_str)
        else:
            if model_key not in LLMS:
                raise ValueError(f"Invalid model key: {model_key}")

            _model = LLMS[model_key]

        self._client_str = _model[4]
        self._model_name = _model[0]
        self._model = _create_model(
            client_str=self._client_str,
            model_name=self._model_name,
            api_key=_model[1],
            base_url=_model[2],
            project=project,
            location=location,
        )

        self.max_interval = _model[3]
        self.mock_res_provider: Callable[..., str] = None

    def model_name(self) -> str:
        return self._model_name

    def chat(self, session_messages, **kwargs):
        if self.mock_res_provider is not None:
            _content = self.mock_res_provider(session_messages, **kwargs)
            res = LLMClientResponse(None)
            res.text = _content
            res.response_token_count = len(_content.split())
            return res

        logging.info("LLM: %s, %s", self.model_name(), kwargs)

        supported = _SUPPORTED_KWARGS.get(self._client_str, set())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported}

        res = LLMClientResponse(None)
        try:
            # Accept either already-built LangChain messages or our dict format.
            if session_messages and isinstance(session_messages[0], BaseMessage):
                lc_messages = session_messages
            else:
                lc_messages = _convert_messages(session_messages)
            model = self._model
            if filtered_kwargs:
                model = model.bind(**filtered_kwargs)
            response = model.invoke(lc_messages)

            res.text = response.content
            res.response = response
            usage = getattr(response, "usage_metadata", None) or {}
            res.prompt_token_count = usage.get("input_tokens", 0)
            res.response_token_count = usage.get("output_tokens", 0)
        except Exception as e:
            res.error = e
            logging.error("LLM: %s, %s", self.model_name(), e)

        return res
