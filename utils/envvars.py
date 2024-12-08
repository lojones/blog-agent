import os
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

# Define constants from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
PPLX_API_KEY = os.getenv('PPLX_API_KEY')
PERSONALITY = os.getenv('PERSONALITY')
LLM_CLAUDE_SONNET = os.getenv('LLM_CLAUDE_SONNET')
LLM_GPT_4O = os.getenv('LLM_GPT_4O')

# Validate required variables
required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GITHUB_TOKEN', 'PPLX_API_KEY', 'PERSONALITY', 'LLM_CLAUDE_SONNET', 'LLM_GPT_4O']
missing_vars = [var for var in required_vars if not globals()[var]]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
