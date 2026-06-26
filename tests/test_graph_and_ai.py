import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.graph_and_ai import process, evaluate, route, agent
from src.schemas import Test_state


def test_process_node_generates_text():
    """Перевіряє, чи нода Письменника додає нове повідомлення в історію."""
    initial_state = {
        "messages": [HumanMessage(content="Напиши короткий слоган для кави.")],
        "feedback": [],
        "approve": False
    }
    
    new_state = process(initial_state)
    
    assert "messages" in new_state
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
    
    updated_state = evaluate(initial_state)
    
    assert "approve" in updated_state
    assert "feedback" in updated_state
    assert isinstance(updated_state["approve"], bool)
    assert isinstance(updated_state["feedback"], list)


@pytest.mark.parametrize("approved_value, expected_route", [
    (True, "finish"),
    (False, "rewrite"),
    (None, "rewrite")  
])
def test_route_logic(approved_value, expected_route):
    """Перевіряє, чи правильно працює логіка умовних переходів."""
    state: Test_state = {
        "messages": [],
        "feedback": [],
        "approve": approved_value
    }
    
    destination = route(state)
    
    assert destination == expected_route

def test_full_agent_execution():
    """Перевіряє роботу всього зібраного графа від початку до кінця."""
    input_data = {
        "messages": [HumanMessage(content="Напиши один рядок тексту без помилок.")]
    }
    

    result = agent.invoke(input_data, config={"recursion_limit": 5})
    
    assert "messages" in result
    assert "approve" in result
    assert "feedback" in result
    
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["approve"] in [True, False]