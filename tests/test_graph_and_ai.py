import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.graph_and_ai import process, evaluate, route, agent
from src.schemas import Test_state, Critic_schema

# =====================================================================
# 1. ЮНІТ-ТЕСТИ ДЛЯ ОКРЕНИХ НОД (UNIT TESTS)
# =====================================================================

def test_process_node_generates_text():
    """Перевіряє, чи нода Письменника додає нове повідомлення в історію."""
    # Готуємо початковий стан (імітуємо вхід від користувача)
    initial_state = {
        "messages": [HumanMessage(content="Напиши короткий слоган для кави.")],
        "feedback": [],
        "approve": False
    }
    
    # Викликаємо суто функцію письменника
    new_state = process(initial_state)
    
    # Перевіряємо результат
    assert "messages" in new_state
    # Оскільки ти використовуєш .append() у своїй версії:
    assert len(new_state["messages"]) > 1
    assert isinstance(new_state["messages"][-1], AIMessage)
    assert len(new_state["messages"][-1].content) > 0


def test_evaluate_node_returns_structured_output():
    """Перевіряє, чи Критик правильно заповнює поляapproved та feedback."""
    initial_state = {
        "messages": [
            HumanMessage(content="Напиши текст."),
            AIMessage(content="Привіт! Це тестовий текст про програмування на Python.")
        ],
        "feedback": [],
        "approve": False
    }
    
    # Викликаємо ноду Критика
    updated_state = evaluate(initial_state)
    
    # Перевіряємо, чи з'явилися потрібні ключі після роботи моделі
    assert "approve" in updated_state
    assert "feedback" in updated_state
    assert isinstance(updated_state["approve"], bool)
    assert isinstance(updated_state["feedback"], list)


@pytest.mark.parametrize("approved_value, expected_route", [
    (True, "finish"),
    (False, "rewrite"),
    (None, "rewrite")  # Якщо раптом ключ порожній
])
def test_route_logic(approved_value, expected_route):
    """Перевіряє, чи правильно працює логіка умовних переходів."""
    state: Test_state = {
        "messages": [],
        "feedback": [],
        "approve": approved_value
    }
    
    # Викликаємо функцію маршрутизатора
    destination = route(state)
    
    # Перевіряємо, чи правильно розгалужується граф
    assert destination == expected_route


# =====================================================================
# 2. ІНТЕГРАЦІЙНИЙ ТЕСТ ДЛЯ ВСЬОГО ГРАФА (INTEGRATION TEST)
# =====================================================================

def test_full_agent_execution():
    """Перевіряє роботу всього зібраного графа від початку до кінця."""
    input_data = {
        "messages": [HumanMessage(content="Напиши один рядок тексту без помилок.")]
    }
    
    # Запускаємо весь граф через invoke
    # Ставимо невеликий ліміт кроків, щоб тест не завис, якщо моделі зацикляться
    result = agent.invoke(input_data, config={"recursion_limit": 5})
    
    # Перевірки фінального стану графа
    assert "messages" in result
    assert "approve" in result
    assert "feedback" in result
    
    # Перевіряємо, що в кінці ми отримали фінальну відповідь від ШІ
    assert isinstance(result["messages"][-1], AIMessage)
    # Якщо граф завершився успішно, approved має бути True (або ліміт кроків зупинив його)
    assert result["approve"] in [True, False]