import logging
from utils.logger import setup_logger
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.github_reader import GithubReader
from enum import Enum
from agent.data_class.blog_data import BlogState, BlogOutlineSimple
from utils.envvars import LLM_CLAUDE_SONNET
from pydantic import BaseModel, Field

logger = setup_logger("BloggerTool")

class WriterTool:
    """Tool for generating blog posts using LLM"""

    class ArticlePart(Enum):
        INTRO = 'INTRO'
        BODY = 'BODY'
        CONCLUSION = 'CONCLUSION'
        FULL_ARTICLE = 'FULL_ARTICLE'
        THESIS = 'THESIS'
        OUTLINE_PREWRITING = 'OUTLINE_PREWRITING'

    system_prompts_github_url = {
        ArticlePart.INTRO: 'https://github.com/lojones/blog-agent-data/blob/main/revise-blog-post-intro.md',
        ArticlePart.BODY: 'https://github.com/lojones/blog-agent-data/blob/main/revise-blog-post-body.md',
        ArticlePart.CONCLUSION: 'https://github.com/lojones/blog-agent-data/blob/main/revise-blog-post-conclusion.md',
        ArticlePart.FULL_ARTICLE: 'https://github.com/lojones/blog-agent-data/blob/main/create-blog-post.md',
        ArticlePart.THESIS: 'https://github.com/lojones/blog-agent-data/blob/main/system-prompts/create-thesis.md',
        ArticlePart.OUTLINE_PREWRITING: 'https://github.com/lojones/blog-agent-data/blob/main/prewriting.md'

    }

    def __init__(self):
        self.logger = logger
        # self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.llm_anthropic = ChatAnthropic(model=LLM_CLAUDE_SONNET, 
                                           temperature=0.7,
                                           max_tokens=8000)
        self.llm_anthropic_structured_outline = self.llm_anthropic.with_structured_output(BlogOutlineSimple, include_raw=True)
        self.github_reader = GithubReader()
        self.logger.info("Initializing BloggerTool - getting system prompts from GitHub")

        self.system_prompts = {
            self.ArticlePart.INTRO: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.INTRO]),
            self.ArticlePart.BODY: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.BODY]),
            self.ArticlePart.CONCLUSION: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.CONCLUSION]),
            self.ArticlePart.FULL_ARTICLE: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.FULL_ARTICLE]),
            self.ArticlePart.THESIS: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.THESIS]),
            self.ArticlePart.OUTLINE_PREWRITING: self.github_reader.read_file(self.system_prompts_github_url[self.ArticlePart.OUTLINE_PREWRITING])
        }

    def construct_thesis(self, instructions: str) -> str:
        """
        Constructs the thesis of the blog post
        """
        self.logger.info(f"Constructing thesis from these instructions: {instructions}")
        system_prompt = self.system_prompts[self.ArticlePart.THESIS]
        prompt = f"""
        Write a thesis for the following instructions:
        {instructions}
        """

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm_anthropic.invoke(messages)
            
            if isinstance(response, AIMessage):
                self.logger.info(f"Usage metadata: {response.usage_metadata}")
                self.logger.info("Successfully constructed thesis")
                self.logger.debug(f"Blog post: {response.content}")
                return response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
        except Exception as e:
            self.logger.error(f"Failed to create blog post: {str(e)}")
            raise

    def create_outline(self, thesis: str, research_content: str, author_personality: str, existing_outline: str, outline_evaluation: str) -> str:
        """
        Creates an outline for the blog post
        """
        self.logger.info("Creating outline")
        system_prompt = self.system_prompts[self.ArticlePart.OUTLINE_PREWRITING]
        prompt = f"""
        Write an outline for the following thesis:
        {thesis}

        The following is the research from Perplexity AI:
        {research_content}

        Write it in the voice of the author described here:
        {author_personality}

        """
        if existing_outline and outline_evaluation:
            self.logger.info("Revising outline from evaluation")
            prompt += f"""
            The following is the existing outline:
            {existing_outline}

            The following is the evaluation of the outline with things to improve:
            {outline_evaluation}
            """
        else:
            self.logger.info("Creating new outline")
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            response = self.llm_anthropic_structured_outline.invoke(messages)
            if isinstance(response['parsed'], BlogOutlineSimple):
                self.logger.info("Successfully created outline")
                self.logger.info(f"Usage metadata: {response['raw'].usage_metadata}")
                self.logger.debug(f"Outline: {response['parsed']}")
                return response['parsed']
            else:
                raise ValueError(f"Unexpected response type: {type(response['parsed'])}")
        except Exception as e:
            self.logger.error(f"Failed to create outline: {str(e)}")
            raise
        

    def create_blog_post(self, state: BlogState) -> str:

        try:
            self.logger.info(f"Writing blog post")
            system_prompt = self.system_prompts[self.ArticlePart.FULL_ARTICLE]
            
            prompt = f"""
            Write a blog post following these specific instructions:

            SHORT TITLE:
            {state.outline.outline.short_title}

            LONG TITLE:
            {state.outline.outline.title}

            INTRO:
            {state.outline.outline.intro}

            BODY:
            {state.outline.outline.body}

            CONCLUSION:
            {state.outline.outline.conclusion}

            WRITE IN THIS STYLE AND VOICE:
            {state.author_personality}

            INCORPORATE THIS BACKGROUND RESEARCH:
            {state.outline.research.content}

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
                self.logger.info(f"Usage metadata: {response.usage_metadata}")
                self.logger.debug(f"Blog post: {response.content}")
                return response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
        except Exception as e:
            self.logger.error(f"Failed to create blog post: {str(e)}")
            raise

    def revise_intro(self, state: BlogState) -> str:
        self.logger.info("Revising article introduction")
        return self.revise(self.ArticlePart.INTRO, state)
        
    def revise_body(self, state: BlogState) -> str:
        self.logger.info("Revising article body")
        return self.revise(self.ArticlePart.BODY, state)
    
    def revise_conclusion(self, state: BlogState) -> str:
        self.logger.info("Revising article conclusion")
        return self.revise(self.ArticlePart.CONCLUSION, state)

    def revise(self, article_part: ArticlePart, state: BlogState) -> str:

        try:
            self.logger.info("Revising blog post")
            system_prompt = self.system_prompts[article_part]
            
            if not system_prompt:
                self.logger.error("Failed to fetch revision prompt from GitHub")
                raise ValueError("Could not fetch revision prompt")
            
            prompt = f"""
            Evaluate and/or revise the {article_part} of the following article:

            CURRENT ARTICLE START:
            {state.article.article_text}
            CURRENT ARTICLE END

            INCORPORATE THIS RESEARCH SUMMARY:
            {state.outline.research.content}
            
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm_anthropic.invoke(messages)
            
            if isinstance(response, AIMessage):
                self.logger.info("Successfully revised introduction")
                self.logger.info(f"Usage metadata: {response.usage_metadata}")
                self.logger.debug(f"Revised article: {response.content}")

                return response.content
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
        except Exception as e:
            self.logger.error(f"Failed to revise introduction: {str(e)}")
            raise


            