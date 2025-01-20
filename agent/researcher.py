from dotenv import load_dotenv
from utils.logger import setup_logger
from agent.tool.authorpersonality import PersonalityTool
from agent.tool.perplexity import PerplexityTool
from agent.tool.writertool import WriterTool
from agent.data_class.blog_data import BlogState, ResearchResponse, BlogOutlineSimple, BlogOutline
from utils.utils import showgraph
from langgraph.graph import StateGraph, START, END
from rich import print
from agent.tool.evaluator import Evaluator
from typing import Literal
from agent.tool.websitecontent import WebsiteContentTool
from utils.utils import write_to_file
load_dotenv()

class Researcher:
    def __init__(self):
        self.name = "Researcher"
        self.logger = setup_logger("Researcher")
        self.logger.info("Initializing Researcher")
        self.personality = PersonalityTool()
        self.perplexity = PerplexityTool()
        self.writer = WriterTool()
        self.evaluator = Evaluator()
        self.websitecontent = WebsiteContentTool()

        self.builder = StateGraph(BlogState)
        self.builder.add_node("create_thesis", self.create_thesis)
        self.builder.add_node("research_thesis", self.research_thesis)
        self.builder.add_node("get_author_personality", self.get_author_personality)
        self.builder.add_node("create_outline", self.create_outline)
        self.builder.add_node("evaluate_outline_quality", self.evaluate_outline_quality)

        self.builder.add_edge(START, "create_thesis")
        self.builder.add_edge("create_thesis", "research_thesis")
        self.builder.add_edge("research_thesis", "get_author_personality")
        self.builder.add_edge("get_author_personality", "create_outline")
        self.builder.add_edge("create_outline", "evaluate_outline_quality")
        self.builder.add_conditional_edges("evaluate_outline_quality", self.is_it_interesting)

        self.graph = self.builder.compile()
        self.logger.info("Successfully initialized LLM and graph")


    def create_thesis(self, state: BlogState) -> BlogState:
        """
        Creates the thesis of the blog post
        """
        self.logger.info("Researcher: Creating thesis")
        instructions = state.article_idea
        thesis = self.writer.construct_thesis(instructions)
        write_to_file(thesis, "thesis", self.logger)
        state.outline.thesis = thesis
        self.logger.info(f"Researcher: Thesis created")
        return state

    def research_thesis(self, state: BlogState) -> BlogState:
        """
        Researches the thesis of the blog post
        """
        self.logger.info("Researcher: Researching thesis")
        thesis = state.outline.thesis
        research : ResearchResponse = self.perplexity.query(thesis)
        write_to_file(research, "research_thesis", self.logger)
        state.outline.research = research
        self.logger.info("Researcher: Thesis researched")
        return state
    
    def get_research_website_content(self, state: BlogState) -> BlogState:
        """
        Gets the content of the research websites
        """
        self.logger.info("Researcher: Getting research website content")
        research_content = self.websitecontent.get_content_from_urls(state.outline.research.sources)
        write_to_file(research_content, "research_content", self.logger)
        state.outline.research.sources_content = research_content
        self.logger.info("Researcher: Research website content retrieved")
        return state
    
    def get_author_personality(self, state: BlogState) -> BlogState:
        """
        Gets the author's personality
        """
        self.logger.info("Researcher: Getting author's personality")
        personality = self.personality.get_author_personality()
        write_to_file(personality, "author_personality", self.logger)
        state.author_personality = personality
        self.logger.info("Researcher: Author's personality retrieved")
        return state
    
    def create_outline(self, state: BlogState) -> BlogState:
        """
        Creates the outline of the blog post
        """
        self.logger.info("Researcher: Creating outline")
        existing_outline = None
        outline_evaluation = None
        if state.outline.outline_evaluation.good_to_go == False:
            existing_outline = state.outline.outline.model_dump_json()
            outline_evaluation = state.outline.outline_evaluation.evaluation
        outline : BlogOutlineSimple = self.writer.create_outline(state.outline.thesis, state.outline.research.content, state.author_personality, existing_outline, outline_evaluation)
        write_to_file(outline, "outline", self.logger)
        state.outline.outline = outline
        self.logger.info("Researcher: Outline created")
        return state
    
    def expose_state(self, state: BlogState) -> BlogState:
        """
        Exposes the state of the blog post
        """
        self.logger.info("Researcher: Exposing state")
        print(state)
        showgraph(self.graph, self.logger, "researcher_graph")
        return state

    def evaluate_outline_quality(self, state: BlogState) -> BlogState:
        """
        Checks if the blog post is interesting
        """
        self.logger.info("Researcher: evaluating the quality of the outline")
        outline : BlogOutline = self.evaluator.evaluate_outline(state.outline)
        write_to_file(outline, "outline_evaluation", self.logger)
        state.outline = outline
        self.logger.info("Researcher: Outline evaluated")
        return state


    def is_it_interesting(self, state: BlogState) -> Literal["create_outline","__end__"]:
        """
        Checks if the blog post is interesting
        """
        self.logger.info("Researcher: decide what to do based on the outline evaluation")
        if state.outline.outline_evaluation.good_to_go or state.outline.outline_evaluation.iteration_number >= 3:
            return END
        else:
            return "create_outline"
