from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from utils.logger import setup_logger
from langgraph.graph import StateGraph, START, END
from agent.tool_planner import PlannerTool

from flask import send_file
import io
from agent.tool_prewriting import PrewritingTool
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from agent.researcher import Researcher
from agent.data_class.blog_data import BlogState

class Supervisor:
    def __init__(self):
        self.name = "Supervisor"
        self.logger = setup_logger("Supervisor")
        self.logger.info("Initializing Supervisor")
        self.researcher = Researcher()
        self.graph = self.build_graph()
        # self.model = ChatOpenAI(model="gpt-4o")
        # self.llm_with_tools = self.model.bind_tools([self.prewriting.derive_prewriting])
        
    def build_graph(self):
        builder = StateGraph(BlogState)
        builder.add_node('research_subgraph', self.researcher.graph)
        builder.add_edge(START, 'research_subgraph')
        builder.add_edge('research_subgraph', END)
        graph = builder.compile()
        return graph


    # def create_plan(self, state: BAState):
    #     self.logger.info("Creating plan")
    #     plan : ExecutionPlan = self.planner.create_execution_plan(state.instructions)
    #     self.logger.log_to_file(f"Plan: {json.dumps(plan.model_dump(),indent=2)}", "plan.json")
    #     state.plan = plan
    #     return state

    # def first_pass(self, state: BAState):
    #     self.logger.info("Creating first pass")
    #     sysprompt: SystemMessage = SystemMessage(content="Perform the functions described using the tools provided") 
    #     prompt = ChatPromptTemplate.from_messages(
    #         [("system", "Perform the functions described using the tools provided"),
    #          ("human", "{input}"),
    #          ("placeholder", "{agent_scratchpad}")]
    #     )
    #     agent = create_tool_calling_agent(self.llm_with_tools, tools=[self.prewriting.derive_prewriting], prompt=prompt)
    #     for step in state.plan.steps:
    #         self.logger.info(f"Step: {step.description}")
    #         response = agent.invoke({"input":step.description,"intermediate_steps": []})
    #         # response = agent.invoke({"input":"test"})
    #         self.logger.info(f"Response: {response}")
    #     return state

    def create_blogpost(self, instructions: str):
        self.logger.info("Creating blogpost")
        input_data = BlogState.model_construct()
        input_data.article_idea = instructions  
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