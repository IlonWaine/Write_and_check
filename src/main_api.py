from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from openai import RateLimitError

# Імпортуємо твого скомпільованого агента з твого файлу
from src.graph_and_ai import agent

app = FastAPI(title="AI Email Writer-Critic Agent")

# Модель для валідації вхідних даних від веб-сторінки
class EmailRequest(BaseModel):
    prompt: str

# 1. Ендпоінт для запуску ШІ-агента
@app.post("/api/generate-email")
async def generate_email(payload: EmailRequest):
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Промпт не може бути порожнім")
        
    try:
        # Готуємо початковий стан для твого LangGraph графа
        conversation_history = [
            HumanMessage(content=f"Write an email based on these details: {payload.prompt}")
        ]
        
        # Запускаємо граф (recursion_limit захищає від нескінченного циклу безкоштовних моделей)
        result = agent.invoke(
            {"messages": conversation_history}, 
            config={"recursion_limit": 4}
        )   
        
        # Беремо вміст самого останнього повідомлення в історії (фінал після правок критика)
        final_email = result["messages"][-1].content
        
        return {"result": final_email}
    
    except GraphRecursionError as e:
        print(f"Досягнуто ліміту рекурсії: {e}")
        
        # МАГІЯ: Дістаємо стан графа на момент зупинки
        # e.state містить останній збережений Test_state
        if hasattr(e, 'state') and e.state:
            last_messages = e.state.get('messages', [])
            if last_messages:
                # Повертаємо останню чернетку від Письменника
                final_email = last_messages[-1].content
                return {
                    "result": final_email,
                    "note": "Увага: Досягнуто ліміту кроків. Повернуто останню версію до фінального схвалення критика."
                }
        
        raise HTTPException(status_code=500, detail="Граф зупинився через ліміт кроків, але стан порожній.")
    except RateLimitError as e:
        # 1. ТУТ ПЕРЕХОПЛЮЄМО ПОМИЛКУ 429 ВІД OPENROUTER
        print(f" Спіймали ліміт OpenRouter: {e}")
        
        # Повертаємо красиву та зрозумілу фразу для фронтенду
        raise HTTPException(
            status_code=429, 
            detail="Вибачте, безкоштовні ліміти запитів до ШІ на сьогодні вичерпано."
        )    

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка роботи агента: {str(e)}")

# 2. Ендпоінт для віддачі головної сторінки сайту
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("src\index.html", "r", encoding="utf-8") as f:
        return f.read()

