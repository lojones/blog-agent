from langgraph.graph import StateGraph, START, END
from agent.data_class.blog_data import BlogState
from agent.tool.writertool import WriterTool
from agent.tool.evaluator import Evaluator
from dotenv import load_dotenv
from utils.logger import setup_logger
from utils.utils import write_to_file
from typing import Literal
load_dotenv()



class Writer:
    def __init__(self):
        self.logger = setup_logger("Writer")
        self.logger.info("Initializing Writer")
        self.writer_tool = WriterTool()
        self.evaluator = Evaluator()
        self.builder = StateGraph(BlogState)
        self.builder.add_node("write_article", self.write_article)
        self.builder.add_node("revise_intro", self.revise_intro)
        self.builder.add_node("revise_body", self.revise_body)
        self.builder.add_node("revise_conclusion", self.revise_conclusion)
        self.builder.add_node("collect_article_parts", self.collect_article_parts)
        self.builder.add_node("evaluate_article", self.evaluate_article)
        self.builder.add_edge(START, "write_article")
        self.builder.add_edge("write_article", "revise_intro")
        self.builder.add_edge("revise_intro", "revise_body")
        self.builder.add_edge("revise_body", "revise_conclusion")
        self.builder.add_edge("revise_conclusion", "collect_article_parts")
        self.builder.add_edge("collect_article_parts", "evaluate_article")
        self.builder.add_conditional_edges("evaluate_article", self.is_it_good_to_go)

        self.graph = self.builder.compile()

    def write_article(self, state: BlogState) -> BlogState:
        self.logger.info("Writing article")
        article = self.writer_tool.create_blog_post(state)
        write_to_file(article, "article_firstpass", self.logger)
        state.article.article_text = article
        self.logger.info(f"Article written")
        return state
    
    def revise_intro(self, state: BlogState) -> BlogState:
        self.logger.info("Revising intro")
        revised_text = self.writer_tool.revise_intro(state)
        write_to_file(revised_text, "article_revised_intro", self.logger)
        state.article.revised_intro_text = revised_text
        self.logger.info(f"Intro revised")
        return state
    
    def revise_body(self, state: BlogState) -> BlogState:
        self.logger.info("Revising body")
        revised_text = self.writer_tool.revise_body(state)
        write_to_file(revised_text, "article_revised_body", self.logger)
        state.article.revised_body_text = revised_text
        self.logger.info(f"Body revised")
        return state

    def revise_conclusion(self, state: BlogState) -> BlogState:
        self.logger.info("Revising conclusion")
        revised_text = self.writer_tool.revise_conclusion(state)
        write_to_file(revised_text, "article_revised_conclusion", self.logger)
        state.article.revised_conclusion_text = revised_text
        
        self.logger.info(f"Conclusion revised")
        return state
    
    def collect_article_parts(self, state: BlogState) -> BlogState:
        self.logger.info("Collecting article parts")
        state.article.article_text_history.append(state.article.article_text)
        state.article.article_text = state.article.revised_intro_text + state.article.revised_body_text + state.article.revised_conclusion_text
        self.logger.info(f"Article parts collected")
        return state
    
    def evaluate_article(self, state: BlogState) -> BlogState:
        self.logger.info("Evaluating article")
        article = self.evaluator.evaluate_article(state.article)
        state.article = article
        self.logger.info(f"Article evaluated")
        return state
    
    def is_it_good_to_go(self, state: BlogState) -> Literal["revise_intro","__end__"]:
        self.logger.info("Deciding if the article is good to go")
        if state.article.article_evaluation.good_to_go or state.article.article_evaluation.iteration_number >= 3:
            return END
        else:
            return "write_article"


