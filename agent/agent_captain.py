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
from agent.tool_websitecontent import WebsiteContentTool
from agent.tool_blogger import BloggerTool
from typing import Dict
from typing_extensions import TypedDict
import json
# from agent.tool_blogger import BlogPostInput
from agent.data_class.blog_data import BlogAgentState
from typing import Literal
from langgraph.prebuilt import tools_condition

load_dotenv()

class AgentCaptain:
    def __init__(self):
        self.name = "Captain"
        self.logger = setup_logger("AgentCaptain")
        self.logger.info("Initializing AgentCaptain")
        
        try:
            self.personality = PersonalityTool()
            self.perplexity = PerplexityTool()
            self.website_content = WebsiteContentTool()
            self.blogger = BloggerTool()
            self.llm = ChatOpenAI(model="gpt-4o")
            self.builder = StateGraph(BlogAgentState)
            self.builder.add_node("initial_outline", self.initial_outline)
            self.builder.add_node("research_background", self.research_background)
            self.builder.add_node("author_personality", self.author_personality)
            self.builder.add_node("unwrap_research_websites", self.unwrap_research_websites)
            self.builder.add_node("write_blogpost", self.write_blogpost)
            self.builder.add_node("revise_blogpost_intro", self.revise_blogpost_intro)
            self.builder.add_node("revise_blogpost_body", self.revise_blogpost_body)
            self.builder.add_node("revise_blogpost_conclusion", self.revise_blogpost_conclusion)
            self.builder.add_node("revise_blogpost", self.revise_blogpost)
            self.builder.add_edge(START, "initial_outline")
            self.builder.add_edge("initial_outline", "research_background")
            self.builder.add_edge("research_background", "author_personality")
            self.builder.add_edge("author_personality", "unwrap_research_websites")
            self.builder.add_edge("unwrap_research_websites", "write_blogpost")
            self.builder.add_edge("write_blogpost", "revise_blogpost")
            self.builder.add_edge("revise_blogpost", "revise_blogpost_intro")
            self.builder.add_edge("revise_blogpost", "revise_blogpost_body")
            self.builder.add_edge("revise_blogpost", "revise_blogpost_conclusion")
            self.builder.add_conditional_edges("revise_blogpost_intro", self.evaluate_revised_blogpost)
            self.builder.add_conditional_edges("revise_blogpost_body", self.evaluate_revised_blogpost)
            self.builder.add_conditional_edges("revise_blogpost_conclusion", self.evaluate_revised_blogpost)
            self.graph = self.builder.compile()
            self.logger.info("Successfully initialized LLM and graph")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM components: {str(e)}")
            raise

    def evaluate_revised_blogpost(self, state: BlogAgentState) -> Literal["revise_blogpost", "__end__"]:
        last_response = state["last_response"]
        if "DONE" in last_response:
            return END
        else:
            return "revise_blogpost"
        
    def revise_blogpost(self, state: BlogAgentState):
        self.logger.info("Revising blogpost")
        return state
    
    def revise_blogpost_intro(self, state: BlogAgentState):
        self.logger.info("Revising blogpost intro")
        blogpost_with_revised_intro = self.blogger.revise_intro(state)
        state["blog_post"] = blogpost_with_revised_intro
        return state
    
    def revise_blogpost_body(self, state: BlogAgentState):
        self.logger.info("Revising blogpost body")
        blogpost_with_revised_body = self.blogger.revise_body(state)
        state["blog_post"] = blogpost_with_revised_body
        return state
    
    def revise_blogpost_conclusion(self, state: BlogAgentState):
        self.logger.info("Revising blogpost conclusion")
        blogpost_with_revised_conclusion = self.blogger.revise_conclusion(state)
        state["blog_post"] = blogpost_with_revised_conclusion
        return state
    
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
            self.logger.log_to_file(state['blog_outline'], "blog_outline.md")
            self.logger.debug(f"Updated blog outline: {state['blog_outline']}")
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(content="**You are a professional content editor and title specialist. Your task is \
                    to take a rough topic statement and transform it into a single, catchy, and \
                        engaging blog post title. Follow these guidelines:** \
                1. **Clarity**: Ensure the title clearly conveys the topic. \
                2. **Engagement**: Use power words, numbers, or intriguing phrasing to grab attention. \
                3. **SEO Optimization**: Include relevant keywords to make the title search-friendly. \
                4. **Conciseness**: Keep the title short and impactful. \
                5. **Tone Alignment**: Match the tone to the target audience and the blogs purpose. \
                ### Example Input: \
                **Rough Topic Statement**: Describing my experience using Cursor AI, \
                    it was good and I thought it was better than GitHub Copilot. I find it hard to \
                        go back to anything else. \
                ### Example Output: \
                *Why Im Sticking with Cursor AI: A Game-Changer Over GitHub Copilot*"))
            response = self.llm.invoke(messages)
            if isinstance(response, AIMessage):
                state["revised_topic"] = response.content
                self.logger.log_to_file(state['revised_topic'], "revised_topic.md")
            else:
                self.logger.warning(f"Received unexpected response type: {type(response)}")
                raise Exception(f"Received unexpected response type: {type(response)}")
        else:
            self.logger.warning(f"Received unexpected response type: {type(response)}")
            raise Exception(f"Received unexpected response type: {type(response)}")
        return state
        

    def research_background(self, state: BlogAgentState):
        topic = state["revised_topic"]
        blog_outline = state["blog_outline"]
        self.logger.info(f"Researching background for topic: {topic}")
        perplexity_response = self.perplexity.query(topic, blog_outline)
        state["background_research_summary"] = perplexity_response
        
        # Convert sources list to JSON string before logging
        sources_json = json.dumps(state['background_research_summary']['sources'], indent=2)
        
        self.logger.log_to_file(state['background_research_summary']['content'], "background_research_summary_content.md")
        self.logger.log_to_file(sources_json, "background_research_summary_sources.md")
        self.logger.debug(f"Updated background research: {state['background_research_summary']}")
        return state

    def author_personality(self, state: BlogAgentState):
        blog_outline = state["blog_outline"]
        topic = state["revised_topic"]
        background_research_summary: ResearchResponse = state["background_research_summary"]
        self.logger.info(f"Analyzing author personality for topic: {topic}")
        background_research_content = background_research_summary["content"]
        author_personality = self.personality.personalize(topic, blog_outline, background_research_content)
        state["author_personality_notes"] = author_personality
        self.logger.log_to_file(state['author_personality_notes'], "author_personality_notes.md")
        return state
    
    def unwrap_research_websites(self, state: BlogAgentState):
        self.logger.info("Unwrapping research websites")
        background_research: ResearchResponse = state["background_research_summary"]
        urls = background_research["sources"]
        for url in urls:
            content = self.website_content.get_content(url)
            if "background_research_content" not in state:
                state["background_research_content"] = content
            else:
                state["background_research_content"] += content
        self.logger.log_to_file(state['background_research_content'], "background_research_content.md")
        return state

        
    def write_blogpost(self, state: BlogAgentState):
        self.logger.info("Writing blogpost")
        blogpost = self.blogger.create_blog_post(state)
        state["blog_post"] = blogpost
        self.logger.log_to_file(state['blog_post'], "blog_post.md")
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

    def create_blogpost(self, instructions: str):
        try:
            # Wrap messages in a dictionary
            input_data: BlogAgentState = {
                "instructions": instructions,
            }
            
            self.logger.info(f"Start Graph with Input data: {input_data}")
            # Pass the dictionary to the graph
            response = self.graph.invoke(input_data)
            return response["blog_post"]
                    
        except Exception as e:
            self.logger.error(f"Error in create_blogpost: {str(e)}")
            raise


