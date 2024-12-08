
from typing_extensions import TypedDict
from typing import Dict, List

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

