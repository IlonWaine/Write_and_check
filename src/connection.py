import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from src.schemas import Critic_schema

load_dotenv()
OPENROUTER_API_KEY = os.getenv('KEY')
# GOOGLE_KEY = os.getenv('GOOGLE_KEY')
API_LINK = "https://openrouter.ai/api/v1"
WRITER_MODEL = "cohere/north-mini-code:free"
CRITIC_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OTHER_ONE = 'openai/gpt-oss-20b:free'


writer_llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base= API_LINK,
    model=WRITER_MODEL, 
    temperature=0.7,
    
)

raw_critic_llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base= API_LINK,
    model=OTHER_ONE, 
    temperature=0.1,
    max_retries=2
)

critic_backup_1 = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=API_LINK,
    model="meta-llama/llama-3.3-70b-instruct:free", 
    temperature=0.1, 
)

critic_backup_2 = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=API_LINK,
    model="google/gemini-flash-1.5:free",
    temperature=0.1
)



try_critic_llm = raw_critic_llm.with_fallbacks([critic_backup_1,critic_backup_2])

critic_llm = try_critic_llm.with_structured_output(Critic_schema)
