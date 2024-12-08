import logging
from utils.logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.github_reader import GithubReader
from enum import Enum
from agent.data_class.blog_data import BlogAgentState
from utils.envvars import LLM_CLAUDE_SONNET

logger = setup_logger("BloggerTool")

class BloggerTool:
    """Tool for generating blog posts using LLM"""

    class ArticlePart(Enum):
        INTRO = 'INTRO'
        BODY = 'BODY'
        CONCLUSION = 'CONCLUSION'
        FULL_ARTICLE = 'FULL_ARTICLE'

    system_prompts_github_url = {
        ArticlePart.INTRO: 'https://github.com/lojones/blog-agent-data/blob/main/revise-blog-post-intro.md',
        ArticlePart.BODY: 'https://github.com/lojones/blog-agent-data/blob/main/revise-blog-post-body.md',
        ArticlePart.CONCLUSION: 'https://github.com/lojones/blog-agent-data/blob/main/revise-blog-post-conclusion.md',
        ArticlePart.FULL_ARTICLE: 'https://github.com/lojones/blog-agent-data/blob/main/create-blog-post.md'
    }

    def __init__(self):
        self.logger = logger
        # self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.llm_anthropic = ChatAnthropic(model=LLM_CLAUDE_SONNET, 
                                           temperature=0.7,
                                           max_tokens=8000)
        self.github_reader = GithubReader()
        self.logger.info("Initializing BloggerTool - getting system prompts from GitHub")

        self.system_prompts = {
            self.ArticlePart.INTRO: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.INTRO]),
            self.ArticlePart.BODY: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.BODY]),
            self.ArticlePart.CONCLUSION: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.CONCLUSION]),
            self.ArticlePart.FULL_ARTICLE: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.FULL_ARTICLE])
        }

    def create_blog_post(self, input_data: BlogAgentState) -> str:
        """
        Creates a blog post using the provided information.
        
        Args:
            input_data: BlogPostInput containing:
                - revised_topic: Main topic of the blog post
                - revised_outline: Structured outline of the post
                - author_personality_notes: Notes about author's style and personality
                - background_research_summary: Summary of research findings
                - background_research_content: Full content from research sources
            
        Returns:
            str: Generated blog post content
        """
        try:
            self.logger.info(f"Writing blog post for topic: {input_data['topic']}")
            system_prompt = self.system_prompt_full_article
            
            prompt = f"""
            Write a blog post following these specific instructions:

            TOPIC:
            {input_data['revised_topic']}

            OUTLINE TO FOLLOW:
            {input_data['revised_outline']}

            WRITE IN THIS STYLE AND VOICE:
            {input_data['author_personality_notes']}

            INCORPORATE THIS RESEARCH SUMMARY:
            {input_data['background_research_summary']['content']}

            DETAILED RESEARCH CONTENT:
            {input_data['background_research_content']}

            IMPORTANT GUIDELINES:
            1. Follow the outline structure exactly
            2. Maintain the specified author's voice and style throughout
            3. Incorporate research naturally, not just copying
            4. Include relevant citations when using specific information
            5. Keep the tone consistent with the blog's objective
            6. Make it engaging and readable
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm_anthropic.invoke(messages)
            
            if isinstance(response, AIMessage):
                self.logger.info("Successfully generated blog post")
                self.logger.debug(f"Blog post: {response.content}")
                return response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
        except Exception as e:
            self.logger.error(f"Failed to create blog post: {str(e)}")
            raise

    def revise_intro(self, input_data: BlogAgentState) -> str:
        self.logger.info("Revising article introduction")
        return self.revise(self.ArticlePart.INTRO, input_data)
        
    def revise_body(self, input_data: BlogAgentState) -> str:
        self.logger.info("Revising article body")
        return self.revise(self.ArticlePart.BODY, input_data)
    
    def revise_conclusion(self, input_data: BlogAgentState) -> str:
        self.logger.info("Revising article conclusion")
        return self.revise(self.ArticlePart.CONCLUSION, input_data)

    def revise(self, article_part: ArticlePart, input_data: BlogAgentState) -> str:

        try:
            self.logger.info("Revising blog post")
            system_prompt = self.system_prompts[article_part]
            
            if not system_prompt:
                self.logger.error("Failed to fetch revision prompt from GitHub")
                raise ValueError("Could not fetch revision prompt")
            
            prompt = f"""
            Evaluate and/or revise the following article:

            CURRENT ARTICLE START:
            {input_data['blog_post']}
            CURRENT ARTICLE END

            The following is additional information about the topic and the author's style and voice:
            TOPIC:
            {input_data['revised_topic']}

            WRITE IN THIS STYLE AND VOICE:
            {input_data['author_personality_notes']}

            INCORPORATE THIS RESEARCH SUMMARY:
            {input_data['background_research_summary']['content']}

            DETAILED RESEARCH CONTENT:
            {input_data['background_research_content']}
            
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm_anthropic.invoke(messages)
            
            if isinstance(response, AIMessage):
                self.logger.info("Successfully revised introduction")
                self.logger.debug(f"Revised article: {response.content}")
                return response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
        except Exception as e:
            self.logger.error(f"Failed to revise introduction: {str(e)}")
            raise


            