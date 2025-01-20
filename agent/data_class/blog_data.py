
from typing_extensions import TypedDict
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage

# class ResearchResponse(BaseModel):
#     content: str = Field(default="", description="The content of the research for writing the article")
#     sources: List[str] = Field(default=[], description="The urls of thesources of the research for writing the article")
#     sources_content: str = Field(default="", description="The content of the research websites in concatenated markdown form")
    
# class BlogAgentState(TypedDict):
#     last_response: str
#     instructions: str
#     initial_topic: str
#     revised_topic: str
#     blog_outline: str
#     revised_outline: str
#     author_personality_notes: str
#     blog_post: str
#     background_research_summary: ResearchResponse
#     background_research_content: str
#     sources: list[str]
#     images: Dict[str, bytes]

# class Plan(BaseModel):
#     """Plan to follow in future"""

#     steps: List[str] = Field(
#         description="different steps to follow, should be in sorted order"
#     )

# class Act(BaseModel):
#     """Action to perform."""

#     action: Plan = Field(
#         description="Action to perform. If you need to further use tools to get the answer, use Plan."
#     )

# class ExecutionStep(BaseModel):
#     step: int = Field(description="The step number, starting at 1")
#     description: str = Field(description="The description of what needs to be done in the step")
#     dependencies: List[str] = Field(description="The dependencies (what needs to be done before this step) of the step")
#     status: str = Field( default="pending", description="any of these statuses represent the status of the step: pending, in_progress, completed, failed") 

# class ExecutionPlan(BaseModel):
#     steps: List[ExecutionStep] = Field(description="The steps to follow to achieve the goal, in sorted order")
#     current_step: int = Field(default=1, description="The current step number thats next in line to execute")
#     notes: str = Field(default="", description="Any notes about the plan")

# class ArticleFraming(TypedDict):
#     topic: str
#     title: str
#     outline: str

# class BAState(BaseModel):
#     messages: List[AnyMessage] = Field(
#         default=[],
#         description="The conversation with the LLM"
#     ) 
#     plan: ExecutionPlan  = Field(
#         default=None,
#         description="Plan to follow to achieve the goal"
#     ) 
#     instructions: str = Field(
#         description="The users original instructions for the what to write in the blog post and how to write it"
#     )

class ResearchResponse(BaseModel):
    content: str = Field(default="", description="The content of the research for writing the article")
    sources: List[str] = Field(default=[], description="The urls of thesources of the research for writing the article")
    sources_content: str = Field(default="", description="The content of the research websites in concatenated markdown form")

class BlogOutlineSimple(BaseModel):
    short_title: str = Field(default="", description="A short and splashy title that will be catchy and attention grabbing for the blog post")
    title: str = Field(default="", description="The full form title of the blog post, this should be a bit longer than the short title and include a subtitle")
    intro: str = Field(default="", description="Point form notes of the intro paragraphs")
    body: str = Field(default="", description="Point form notes of the body paragraphs")
    conclusion: str = Field(default="", description="Point form notes of the conclusion paragraphs")

class BlogOutlineEvaluation(BaseModel):
    evaluation: str = Field(default="", description="Detailed evaluation on how to improve the outline of the blog post or article")
    good_to_go: bool = Field(default=True, description="Whether the outline is good to go (true) or needs more work (false)")
    iteration_number: int = Field(default=0, description="The iteration number of the outline, starting at 0")
    messages: List[AnyMessage] = Field(default=[], description="The prompt messages from the evaluator LLM")

class BlogOutline(BaseModel):
    thesis: str = Field(default="", description="The thesis of the blog post")
    research: ResearchResponse = Field(default=None,description="The research response from Perplexity AI")
    outline: BlogOutlineSimple = Field(default=BlogOutlineSimple().model_construct(), description="The core outline points of the blog post")
    outline_evaluation: BlogOutlineEvaluation = Field(default=BlogOutlineEvaluation().model_construct(), description="Notes about how interesting this outline is and notes on improving it (if needed)")

class BlogArticleEvaluation(BaseModel):
    evaluation: str = Field(default="", description="Detailed evaluation on how interesting this article is and notes on improving it (if needed)")
    good_to_go: bool = Field(default=True, description="Whether the article is good to go (true) or needs more work (false)")
    iteration_number: int = Field(default=0, description="The iteration number of the article, starting at 0")
    messages: List[AnyMessage] = Field(default=[], description="The prompt messages from the evaluator LLM")

class BlogArticle(BaseModel):
    article_text: str = Field(default="", description="The raw text of the article or blog post")
    revised_intro_text: str = Field(default="", description="The revised text of the intro paragraph")
    revised_body_text: str = Field(default="", description="The revised text of the body paragraphs")
    revised_conclusion_text: str = Field(default="", description="The revised text of the conclusion paragraph")
    article_evaluation: BlogArticleEvaluation = Field(default=BlogArticleEvaluation().model_construct(), description="Notes about how interesting this article is and notes on improving it (if needed)")

class BlogState(BaseModel):
    article_idea: str = Field(default="", description="The users original instructions for the what to write in the blog post and how to write it")
    outline: BlogOutline = Field(default=BlogOutline().model_construct(), description="The outline points of the blog post")
    author_personality: str = Field(default="", description="A description of the author's personality and writing style")
    article: BlogArticle = Field(default=BlogArticle().model_construct(), description="The article")


