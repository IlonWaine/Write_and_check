
from langgraph.graph import StateGraph, END
from langchain_core.messages import  AIMessage

from src.schemas import Test_state
from src.connection import writer_llm, critic_llm


def process(state: Test_state)-> Test_state:
    
    if state.get('feedback'):
        feedback_text = "\n- ".join(state['feedback'])
        instruction = f"Критик відхилив твій варіант. Виправ наступні помилки:\n- {feedback_text}"
        state['messages'].append(AIMessage(content=instruction))
        print(" Письменник отримав правки від Критика.")

    user_request = f'виконай завдання, не додавай спец символів якщо прямо не попросять {state['messages']}'
    response =  writer_llm.invoke(user_request)
    state['messages'].append(AIMessage(content = response.content))
    print(f'responce {response.content} ')
    return state

def evaluate(state:Test_state)->Test_state:
    print("[Критик] Оцінюю якість тексту...")
    generated_message = state['messages'][-1].content
    eval_promt = f'''ти критик оціниєщ точність та якість {generated_message}
                    відповіді. Якщо немає конкретних даних дадай шаблон по прикладу:
                    [Відправник / Отримувач / Зазначте свою дату], якщо немає потреби
                    вносити конкретні дані то не додавай жодних спец символів  '''
    feedback = critic_llm.invoke(eval_promt)
    state['approve'] = feedback.approved
    state['feedback'] = feedback.suggestions
    print(f"Чи схвалено: {feedback.approved}")
    print(f"Зауваження: {feedback.suggestions}")

  
    return state

def route(state:Test_state)->str:
    if state.get('approve') is True:
        print(" Текст ідеальний! Завершуємо роботу.")
        return 'finish'
    else:
        print("Щодо тексту Є зауваження. Відправляю на переписування.")
        return 'rewrite'

graph = StateGraph(Test_state)

graph.add_node('process',process)
graph.set_entry_point('process')
graph.add_node('eval',evaluate)
graph.add_edge('process','eval')
graph.add_conditional_edges(
    'eval',
    route,
    {
        'finish': END,
        'rewrite': 'process'
    })
agent = graph.compile()  

# conversation_history = []

# try:
#     conversation_history.append(HumanMessage(content='write short email about pay raise, with mistakes unless instracted otherwise'))
#     result = agent.invoke({'messages':conversation_history},config={'recursion_limit':5})
#     print(f'Here result messages {result['messages']}')
#     conversation_history = result['messages']
#     # user_input = input('enter input')
# except Exception as e:
#     print(f"Error encountered: {e}")




