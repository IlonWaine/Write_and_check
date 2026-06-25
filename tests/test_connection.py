# test_connection.py
import os
from src.log_logic import logger
from src.connection import writer_llm, critic_llm
from src.schemas import Critic_schema

def test_writer_connection():
    logger.info("⚡ Перевірка підключення Письменника (writer_llm)...")
    try:
        # Простий тест на звичайний текст
        response = writer_llm.invoke("Привіт! Дай відповідь одним словом 'Успіх'.")
        logger.success("✅ Письменник відповів успішно!")
        logger.info(f"Відповідь моделі: {response.content.strip()}")
        return True
    except Exception as e:
        logger.error(f"❌ Помилка підключення Письменника: {e}")
        return False

def test_critic_structured_connection():
    logger.info("⚡ Перевірка підключення Критика (critic_llm) + Structured Output...")
    try:
        # Тестове повідомлення у форматі списку словників (як ми зафіксили раніше)
        test_messages = [
            {"role": "system", "content": "Ти помічник. Твоє завдання повернути структурований JSON."},
            {"role": "user", "content": "Перевір фразу 'Я люблю кодить'"}
        ]
        
        # Виклик моделі з очікуванням на Pydantic структуру
        result: Critic_schema = critic_llm.invoke(test_messages)
        
        logger.success("✅ Критик відповів успішно! Structured Output працює.")
        logger.info("Згенерований структурований результат:")
        logger.info(f" - Approved (Дозволено): {result.approved}")
        logger.info(f" - Suggestions (Зауваження): {result.suggestions}")
        return True
    except Exception as e:
        logger.error(f"❌ Помилка Критика або структурованого виводу: {e}")
        logger.warning("💡 Порада: якщо впав саме Критик, модель 'cohere/north-mini-code:free' "
                       "може погано підтримувати функцію інструментів (Tool Calling). "
                       "Спробуй змінити модель для Критика на 'meta-llama/llama-3.3-70b-instruct:free'.")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🏁 ПОЧАТОК ТЕСТУВАННЯ З'ЄДНАННЯ З OPENROUTER 🏁")
    print("=" * 60)
    
    # Перевіримо взагалі наявність ключа в системі
    key = os.getenv('KEY')
    if not key:
        logger.critical("❌ Помилка: Змінна оточення 'KEY' порожня! Перевір свій .env файл.")
    else:
        logger.info(f"Ключ знайдено (довжина: {len(key)} символів). Починаємо запити...")
        
        writer_ok = test_writer_connection()
        print("-" * 60)
        critic_ok = test_critic_structured_connection()
        
        print("=" * 60)
        if writer_ok and critic_ok:
            logger.success("🎉 ВСІ ТЕСТИ ПРОЙДЕНО! Можна запускати основний граф (main.py).")
        else:
            logger.error("🛑 Тестування завершилося з помилками. Виправ конфіг перед запуском графа.")
    print("=" * 60)