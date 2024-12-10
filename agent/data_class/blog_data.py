
from typing_extensions import TypedDict
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage

class ResearchResponse(TypedDict):
    content: str
    sources: List[str]
    
class BlogAgentState(TypedDict):
    last_response: str
    instructions: str
    initial_topic: str
    revised_topic: str
    blog_outline: str
    revised_outline: str
    author_personality_notes: str
    blog_post: str
    background_research_summary: ResearchResponse
    background_research_content: str
    sources: list[str]
    images: Dict[str, bytes]

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Act(BaseModel):
    """Action to perform."""

    action: Plan = Field(
        description="Action to perform. If you need to further use tools to get the answer, use Plan."
    )


class BAState(BaseModel):
    messages: List[AnyMessage] = Field(
        default=[],
        description="The conversation with the LLM"
    ) 
    plan: Plan  = Field(
        default=None,
        description="Plan to follow to achieve the goal"
    ) 
    instructions: str = Field(
        description="The users original instructions for the what to write in the blog post and how to write it"
    )

class ExecutionStep(BaseModel):
    step: str
    description: str
    dependencies: List[str]
    status: str  # 'pending', 'in_progress', 'completed', 'failed'

class ExecutionPlan(BaseModel):
    steps: List[ExecutionStep]
    current_step: int
    notes: str