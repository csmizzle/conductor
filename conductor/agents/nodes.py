"""
Nodes for LangGraph
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import FunctionMessage
from conductor.agents import tools, signatures
import dspy
import os
from loguru import logger


def agent(state):
    """
    Invokes the agent model
    """
    logger.info("Invoking agent model ...")
    messages = state["messages"]
    model = ChatOpenAI(
        model="gpt-4o",
        openai_api_base=os.getenv("LITELLM_HOST"),
        openai_api_key=os.getenv("LITELLM_API_KEY"),
        temperature=0,
    )
    model = model.bind_tools([tools.get_document])
    response = model.invoke(messages)
    print("Tool response")
    print(response)
    return {"messages": [response]}


def rewrite(state):
    logger.info("Rewriting query ...")
    messages = state["messages"]
    question = messages[0].content
    question_rewriter = dspy.ChainOfThought(signatures.QuestionRewriter)
    rewritten_question = question_rewriter(question=question)
    return {
        "messages": [
            FunctionMessage(
                content=rewritten_question.rewritten_question, name="rewrite"
            )
        ]
    }


def generate(state):
    logger.info("Generating response ...")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    doc = last_message.content
    # generate answer
    answer = dspy.ChainOfThought(signatures.AnswerGenerator)
    generated_answer = answer(answer=question, document=doc)
    return {
        "messages": [FunctionMessage(content=generated_answer.answer, type="function")]
    }
