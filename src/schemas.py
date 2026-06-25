from typing import List, TypedDict, Union
from pydantic import BaseModel, Field

from langchain.messages import HumanMessage,AIMessage

class AgentState(BaseModel):
    prompt: str
    current_text: str = ""
    feedback: List[str] = Field(default_factory=list)
    is_ok: bool = False
    revision_count: int = 0
    max_revisions: int = 3

class Critic_schema(BaseModel):
    approved: bool = Field(description="True, якщо текст ідеальний і не потребує правок. False, якщо є помилки.")
    suggestions: List[str] = Field(description="Список конкретних зауважень та помилок.")


class Test_state(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]
    feedback: List[str]
    approve: bool