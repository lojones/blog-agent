from dotenv import load_dotenv
from utils.logger import setup_logger
from langgraph.graph import StateGraph, START, END
from agent.tool_planner import PlannerTool
from agent.data_class.blog_data import BAState, ExecutionPlan
from flask import send_file
import io

class Supervisor:
    def __init__(self):
        self.planner = PlannerTool()
        self.name = "Supervisor"
        self.logger = setup_logger("Supervisor")
        self.logger.info("Initializing Supervisor")
        self.graph = self.build_graph()

    def build_graph(self):
        builder = StateGraph(BAState)
        builder.add_node('planner', self.create_plan)
        builder.add_edge(START, 'planner')
        builder.add_edge('planner', END)
        graph = builder.compile()
        return graph

    def create_plan(self, state: BAState):
        self.logger.info("Creating plan")
        plan : ExecutionPlan = self.planner.create_execution_plan(state.instructions)
        state.plan = plan
        return state

    def create_blogpost(self, instructions: str):
        self.logger.info("Creating blogpost")
        input_data: BAState = {
            "instructions": instructions,
        }
        state = self.graph.invoke(input_data)
        return state['plan']
    
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