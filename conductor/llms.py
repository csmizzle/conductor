"""
Implementation of the LLM services
"""
from langchain_community.llms.openai import OpenAI


openai_gpt_4 = OpenAI(model="gpt-4-32k", temperature=0)
