import os, getpass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from utils.logger import setup_logger
from flask import send_file
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import io

load_dotenv()

class AgentCaptain:
    def __init__(self):
        self.name = "Captain"
        self.logger = setup_logger("AgentCaptain")
        self.logger.info("Initializing AgentCaptain")
        
        try:
            self.llm = ChatOpenAI(model="gpt-4o")
            self.builder = StateGraph(MessagesState)
            self.builder.add_node("llm",self.call_llm)
            self.builder.add_edge(START, "llm")
            self.builder.add_edge("llm", END)
            self.graph = self.builder.compile()
            self.logger.info("Successfully initialized LLM and graph")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM components: {str(e)}")
            raise

    def call_llm(self, state: MessagesState):
        return {"messages": [self.llm.invoke(state["messages"])]}
        
    def multiply(self,a: int, b: int):
        """Multiply a and b."""
        self.logger.debug(f"Multiplying {a} * {b}")
        result = a * b
        self.logger.debug(f"Result: {result}")
        return result
    
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



    def create_blogpost(self, topic: str):
        try:
            # Wrap messages in a dictionary

            input_data = { "messages":[HumanMessage(content=f"Create a blog post about {topic}")]}
            
            # Pass the dictionary to the graph
            response = self.graph.invoke(input_data)
            
            # Process the response
            last_message = response["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
            else:
                return last_message["content"]
                    
        except Exception as e:
            self.logger.error(f"Error in create_blogpost: {str(e)}")
            raise
