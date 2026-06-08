import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from llamevol.llm import LLMmanager


@pytest.fixture()
def api_key():
    api_key = "api_keyapi_keyapi_key"
    return api_key


def test_google(api_key):
    llm_manager = LLMmanager(
        model_name="gemini-2.0-flash", api_key=api_key, client_str="google"
    )
    assert isinstance(llm_manager._model, ChatGoogleGenerativeAI)
    assert llm_manager.model_name() == "gemini-2.0-flash"

    llm_manager = LLMmanager(
        model_name="gemini-2.5-flash", api_key=api_key, client_str="google"
    )
    assert llm_manager.model_name() == "gemini-2.5-flash"


def test_vertex(api_key):
    llm_manager = LLMmanager(
        model_name="gemini-2.0-flash", api_key=api_key, client_str="vertex"
    )
    assert isinstance(llm_manager._model, ChatVertexAI)
    assert llm_manager.model_name() == "gemini-2.0-flash"

    llm_manager = LLMmanager(
        model_name="gemini-2.5-flash", api_key=api_key, client_str="vertex"
    )
    assert llm_manager.model_name() == "gemini-2.5-flash"


def test_openai(api_key):
    llm_manager = LLMmanager(model_name="gpt-5", api_key=api_key, client_str="openai")
    assert isinstance(llm_manager._model, ChatOpenAI)
    assert llm_manager.model_name() == "gpt-5"


def test_default_client(monkeypatch, api_key):
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    llm_manager = LLMmanager(model_key="gpt-4o")
    assert isinstance(llm_manager._model, ChatOpenAI)
    assert llm_manager.model_name() == "gpt-4o-2024-11-20"


def test_open_router(api_key):
    llm_manager = LLMmanager(
        model_name="gpt-4o-mini", api_key=api_key, client_str="openrouter"
    )
    assert isinstance(llm_manager._model, ChatOpenAI)
    assert llm_manager.model_name() == "gpt-4o-mini"


def test_request(api_key):
    llm_manager = LLMmanager(
        model_name="claude-sonnet-4.5", api_key=api_key, client_str="request"
    )
    assert isinstance(llm_manager._model, ChatOpenAI)
    assert llm_manager.model_name() == "claude-sonnet-4.5"
