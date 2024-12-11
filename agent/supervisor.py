from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.logger import setup_logger
from langgraph.graph import StateGraph, START, END
from agent.tool_planner import PlannerTool
from agent.data_class.blog_data import BAState, ExecutionPlan
from flask import send_file
import io
from agent.tool_prewriting import PrewritingTool
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

class Supervisor:
    def __init__(self):
        self.planner = PlannerTool()
        self.prewriting = PrewritingTool()
        self.name = "Supervisor"
        self.logger = setup_logger("Supervisor")
        self.logger.info("Initializing Supervisor")
        self.graph = self.build_graph()
        self.model = ChatOpenAI(model="gpt-4o")
        self.llm_with_tools = self.model.bind_tools([self.prewriting.derive_prewriting])
        

    def build_graph(self):
        builder = StateGraph(BAState)
        builder.add_node('planner', self.create_plan)
        builder.add_node('first_pass', self.first_pass)
        builder.add_edge(START, 'planner')
        builder.add_edge('planner', 'first_pass')
        builder.add_edge('first_pass', END)
        graph = builder.compile()
        return graph

    def create_plan(self, state: BAState):
        self.logger.info("Creating plan")
        plan : ExecutionPlan = self.planner.create_execution_plan(state.instructions)
        state.plan = plan
        return state

    def first_pass(self, state: BAState):
        self.logger.info("Creating first pass")
        sysprompt: SystemMessage = SystemMessage(content="Perform the functions described using the tools provided") 
        prompt = ChatPromptTemplate.from_messages(
            [("system", "Perform the functions described using the tools provided"),
             ("human", "{input}"),
             ("placeholder", "{agent_scratchpad}")]
        )
        agent = create_tool_calling_agent(self.llm_with_tools, tools=[self.prewriting.derive_prewriting], prompt=prompt)
        for step in state.plan.steps:
            response = agent.invoke({"input":step.description})
            self.logger.info(f"Response: {response}")
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