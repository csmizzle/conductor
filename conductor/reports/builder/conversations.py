"""
Generate a conversation that builds unique personas and more questions based on inputted research questions
- Read in the research questions
- Create a conversation between the researcher and the writer
"""
import dspy
from conductor.reports.builder import signatures
from conductor.reports.builder import models
from conductor.flow.retriever import ElasticRMClient
from conductor.flow.rag import CitationRAG


class SimulatedConversation(dspy.Module):
    def __init__(
        self,
        retriever: ElasticRMClient,
        max_conversation_turns: int = 3,
    ) -> None:
        self.max_conversation_turns = max_conversation_turns
        self.conversation_topic = dspy.ChainOfThought("input:str -> topic:str")
        self.conversation_turn = dspy.ChainOfThought(signatures.ConversationTurn)
        self.researcher_response = dspy.ChainOfThought(signatures.ResearcherResponse)
        self.refined_question = dspy.ChainOfThought(signatures.RefinedQuestion)
        self.retriever = CitationRAG(elastic_retriever=retriever)

    def forward(self, input_: str) -> dspy.Prediction:
        conversation_history = []
        input_support = None
        conversation_topic = self.conversation_topic(input=input_).topic
        for _ in range(self.max_conversation_turns):
            response = self.conversation_turn(
                topic=conversation_topic,
                conversation_history=conversation_history,
                input_support=input_support,
                input=input_,
            ).response
            conversation_history.append(
                models.Interaction(
                    input=input_, input_support=input_support, response=response
                )
            )
            # get documents based on expert response
            input_support = self.retriever(response)
            input_ = self.researcher_response(
                topic=conversation_topic,
                conversation_history=conversation_history,
                input=input_,
                response=response,
                input_support=input_support,
            ).new_input
        generated_refined_question = self.refined_question(
            topic=conversation_topic,
            conversation_history=conversation_history,
            input=input_,
        ).refined_question
        return dspy.Prediction(
            input=input_,
            topic=conversation_topic,
            conversation_history=conversation_history,
            refined_question=generated_refined_question,
        )
