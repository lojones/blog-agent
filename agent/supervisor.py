from utils.logger import setup_logger
from langgraph.graph import StateGraph, START, END
from flask import send_file
import io
from agent.researcher import Researcher
from agent.writer import Writer
from agent.data_class.blog_data import BlogState
from utils.utils import showgraph

class Supervisor:
    def __init__(self):
        self.name = "Supervisor"
        self.logger = setup_logger("Supervisor")
        self.logger.info("Initializing Supervisor")
        self.researcher = Researcher()
        self.writer = Writer()
        self.graph = self.build_graph()
        
    def build_graph(self):
        builder = StateGraph(BlogState)
        builder.add_node('research_subgraph', self.researcher.graph)
        builder.add_node('write_subgraph', self.writer.graph)
        builder.add_edge(START, 'research_subgraph')
        builder.add_edge('research_subgraph', 'write_subgraph')
        builder.add_edge('write_subgraph', END)
        graph = builder.compile()
        showgraph(graph, self.logger, "supervisor_graph")
        return graph

    def create_blogpost(self, instructions: str):
        self.logger.info("Creating blogpost")
        input_data = BlogState.model_construct()
        input_data.article_idea = instructions  
        state = self.graph.invoke(input_data)
        return state
    
    def showgraph(self):
        try:
            img_data = self.graph.get_graph().draw_mermaid_png()
            img_io = io.BytesIO(img_data)
            img_io.seek(0)
            return send_file(
                img_io,
                mimetype='image/png'
            )
        except Exception as e:
            self.logger.error(f"Failed to display graph: {str(e)}")
            raise