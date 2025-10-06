import pytest
from llamevol.llm import LLMmanager, GoogleGenAIClient, OpenAIClient, RequestClient, AISuiteClient
# 'groq', 'google', 'openai', 'openrouter', 'request', or None for default handling like AISuiteClient

@pytest.fixture()
def api_key():
    api_key = 'api_keyapi_keyapi_key'
    return api_key

def test_google(api_key):
    llm_manager = LLMmanager(model_name='gemini-2.0-flash', api_key=api_key, client_str='google')
    assert isinstance(llm_manager.client, GoogleGenAIClient)
    assert llm_manager.client.name == 'gemini-2.0-flash'
    assert llm_manager.model_name() == 'gemini-2.0-flash'
    assert llm_manager.client.api_key == api_key

    llm_manager = LLMmanager(model_name='gemini-2.5-flash', api_key=api_key, client_str='google')
    assert llm_manager.client.name == 'gemini-2.5-flash'
    assert llm_manager.model_name() == 'gemini-2.5-flash'

def test_vertex(api_key):
    llm_manager = LLMmanager(model_name='gemini-2.0-flash',api_key=api_key, client_str='vertex')
    assert isinstance(llm_manager.client, GoogleGenAIClient)
    assert llm_manager.client.name == 'gemini-2.0-flash'
    assert llm_manager.model_name() == 'gemini-2.0-flash'
    assert llm_manager.client.api_key == api_key

    llm_manager = LLMmanager(model_name='gemini-2.5-flash', api_key=api_key, client_str='vertex')
    assert llm_manager.client.name == 'gemini-2.5-flash'
    assert llm_manager.model_name() == 'gemini-2.5-flash'

def test_openai(api_key):
    llm_manager = LLMmanager(model_name='gpt-5', api_key=api_key, client_str='openai')
    assert isinstance(llm_manager.client, OpenAIClient)
    assert llm_manager.client.name == 'gpt-5'
    assert llm_manager.model_name() == 'gpt-5'
    assert llm_manager.client.api_key == api_key

def test_default_client(monkeypatch, api_key):
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    llm_manager = LLMmanager(model_key='gpt-4o')
    assert isinstance(llm_manager.client, OpenAIClient)
    assert llm_manager.client.name == 'gpt-4o-2024-11-20'
    assert llm_manager.model_name() == 'gpt-4o-2024-11-20'

def test_open_router(api_key):
    llm_manager = LLMmanager(model_name='gpt-4o-mini', api_key=api_key, client_str='openrouter')
    assert isinstance(llm_manager.client, OpenAIClient)
    assert llm_manager.client.name == 'gpt-4o-mini'
    assert llm_manager.model_name() == 'gpt-4o-mini'
    assert llm_manager.client.api_key == api_key

def test_request(api_key):
    llm_manager = LLMmanager(model_name='claude-sonnet-4.5', api_key=api_key, client_str='request')
    assert isinstance(llm_manager.client, RequestClient)
    assert llm_manager.client.name == 'claude-sonnet-4.5'
    assert llm_manager.model_name() == 'claude-sonnet-4.5'
    assert llm_manager.client.api_key == api_key

