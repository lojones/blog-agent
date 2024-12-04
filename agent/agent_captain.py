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
from agent.tool_perplexity import PerplexityTool, ResearchResponse
from agent.tool_authorpersonality import PersonalityTool
from typing import Dict
from typing_extensions import TypedDict

load_dotenv()

class BlogAgentState(TypedDict):
    current_message: str
    past_messages: list[str]
    initial_topic: str
    blog_outline: str
    revised_outline: str
    author_personality_notes: str
    blog_post: str
    background_research: ResearchResponse
    sources: list[str]
    images: Dict[str, bytes]
    

class AgentCaptain:
    def __init__(self):
        self.name = "Captain"
        self.logger = setup_logger("AgentCaptain")
        self.logger.info("Initializing AgentCaptain")
        
        try:
            self.personality = PersonalityTool()
            self.perplexity = PerplexityTool()
            self.llm = ChatOpenAI(model="gpt-4o")
            # self.tools = [self.call_perplexity, self.get_personality_prompt]
            # self.llm_with_tools = self.llm.bind_tools(self.tools)
            self.builder = StateGraph(BlogAgentState)
            self.builder.add_node("initial_outline", self.initial_outline)
            self.builder.add_node("research_background", self.research_background)
            self.builder.add_node("author_personality", self.author_personality)
            self.builder.add_edge(START, "initial_outline")
            self.builder.add_edge("initial_outline", "research_background")
            self.builder.add_edge("research_background", "author_personality")
            self.builder.add_edge("author_personality", END)
            self.graph = self.builder.compile()
            self.logger.info("Successfully initialized LLM and graph")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM components: {str(e)}")
            raise

        

    
    def initial_outline(self, state: BlogAgentState):
        initial_topic = state["initial_topic"]
        self.logger.info(f"Creating initial outline for topic: {initial_topic}")
        messages = []
        sysmsg = "You are a helpful assistant that creates blog posts.  \
            You look at the topic provided and create an initial outline for the blog post, understanding \
                that new information will be provided and you may updated the outline.\
            Follow these rules strictly to create the outline:\
             * Define the title in the first line, it should look like: 'TITLE: <title>' \
             * Decide if the blog post is meant to be informing, persuading, or entertaining then write in the second \
                line, it should look like: 'OBJECTIVE: <informing|persuading|entertaining> \
             * Decide the main point that readers should take away and write that on the third line, \
                it should look like: 'THESIS: <thesis>' \
             * List the key points in point form on the following lines, each one starting with '* <point>' \
             * Outline the structure of the post on the next lines, it should look like: \
                STRUCTURE:\\nINTRO: <intro point>\\nBODY: <4 body points>\\nCONCLUSION: <conclusion point>"
        messages.append(SystemMessage(content=sysmsg))
        messages.append(HumanMessage(content=f"Create an outline for a blog post about this topic: '{initial_topic}'"))
        response = self.llm.invoke(messages)
        if isinstance(response, AIMessage):
            state["blog_outline"] = response.content
            self.logger.info(f"Updated blog outline: {state['blog_outline']}")
        else:
            self.logger.warning(f"Received unexpected response type: {type(response)}")
            state["blog_outline"] = ""
        return state
        
    def research_background(self, state: BlogAgentState):
        initial_topic = state["initial_topic"]
        blog_outline = state["blog_outline"]
        self.logger.info(f"Researching background for topic: {initial_topic}")
        perplexity_response = self.perplexity.query(initial_topic, blog_outline)
        state["background_research"] = perplexity_response
        self.logger.info(f"Updated background research: {state['background_research']}")
        return state

    def author_personality(self, state: BlogAgentState):
        blog_outline = state["blog_outline"]
        initial_topic = state["initial_topic"]
        background_research: ResearchResponse = state["background_research"]
        self.logger.info(f"Analyzing author personality for topic: {initial_topic}")
        background_research_content = background_research["content"]
        author_personality = self.personality.personalize(initial_topic, blog_outline, background_research_content)
        state["author_personality_notes"] = author_personality
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

    def create_blogpost(self, topic: str):
        try:
            # Wrap messages in a dictionary
            input_data: BlogAgentState = {
                "initial_topic": topic,
                "blog_outline": "",
                "background_research": "",
                "author_personality_notes": {}
            }
            
            self.logger.info(f"Start Graph with Input data: {input_data}")
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


