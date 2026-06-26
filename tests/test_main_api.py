import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from openai import RateLimitError


from src.main_api import app

client = TestClient(app)

def test_generate_email_empty_prompt():
    """Перевіряє, що API повертає 400 Bad Request, якщо передано порожній промпт."""
    response = client.post("/api/generate-email", json={"prompt": "   "})
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Промпт не може бути порожнім"

@patch("src.main_api.agent.invoke")
def test_generate_email_success(mock_invoke):
    """Перевіряє успішний сценарій, коли граф відпрацював штатно."""
    mock_invoke.return_value = {
        "messages": [AIMessage(content="Hello! This is a generated email content.")]
    }
    
    #  запит до FastAPI ендпоінту
    response = client.post("/api/generate-email", json={"prompt": "Write a job offer"})
    
    # Перевірка результати
    assert response.status_code == 200
    assert response.json() == {"result": "Hello! This is a generated email content."}
    
    # Перевірка, виклику агента 1 раз із правильними параметрами
    mock_invoke.assert_called_once()

@patch("src.main_api.agent.invoke")
def test_generate_email_recursion_limit_with_state(mock_invoke):
    """Перевіряє, що у разі ліміту кроків повертається остання чернетка."""
    fake_error = GraphRecursionError("Recursion limit reached")
    fake_error.state = {
        "messages": [AIMessage(content="Draft before recursion limit hit.")]
    }
    
    mock_invoke.side_effect = fake_error
    
    response = client.post("/api/generate-email", json={"prompt": "Write something long"})
    
    assert response.status_code == 200
    assert response.json()["result"] == "Draft before recursion limit hit."
    assert "Досягнуто ліміту кроків" in response.json()["note"]

@patch("src.main_api.agent.invoke")
def test_generate_email_rate_limit(mock_invoke):
    """Перевіряє перехоплення помилки 429 від платформи OpenRouter/OpenAI."""

    fake_response = MagicMock()
    fake_response.status_code = 429
    fake_error = RateLimitError("Rate limit exceeded", response=fake_response, body=None)
    
    mock_invoke.side_effect = fake_error
    
    response = client.post("/api/generate-email", json={"prompt": "Generate email"})
    
    assert response.status_code == 429
    assert response.json()["detail"] == "Вибачте, безкоштовні ліміти запитів до ШІ на сьогодні вичерпано."

@patch("builtins.open", new_callable=MagicMock)
def test_get_index_html(mock_open):
    """Перевіряє, чи ендпоінт '/' успішно зчитує та повертає HTML сторінку."""
    mock_file = MagicMock()
    mock_file.read.return_value = "<html><body>Fake Index Page</body></html>"
    mock_open.return_value.__enter__.return_value = mock_file
    
    response = client.get("/")
    
    assert response.status_code == 200
    assert "Fake Index Page" in response.text