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
        max_conversation_turns: int = 5,
    ) -> None:
        self.max_conversation_turns = max_conversation_turns
        self.conversation_turn = dspy.ChainOfThought(signatures.ConversationTurn)
        self.researcher_response = dspy.ChainOfThought(signatures.ResearcherResponse)
        self.refined_question = dspy.ChainOfThought(signatures.RefinedQuestion)
        self.retriever = CitationRAG(elastic_retriever=retriever)

    def forward(self, input_: str) -> dspy.Prediction:
        conversation_history = []
        documents = []
        for _ in range(self.max_conversation_turns):
            response = self.conversation_turn(
                conversation_history=conversation_history,
                supporting_documents=documents,
                input=input_,
            ).response
            conversation_history.append(
                models.Interaction(
                    input=input_, supporting_documents=documents, response=response
                )
            )
            # get documents based on expert response
            documents = self.retriever(response)
            input_ = self.researcher_response(
                conversation_history=conversation_history,
                input=input_,
                response=response,
                supporting_documents=documents,
            ).new_input
        generated_refined_question = self.refined_question(
            conversation_history=conversation_history, input=input_
        ).refined_question
        return dspy.Prediction(
            input=input_,
            conversation_history=conversation_history,
            refined_question=generated_refined_question,
        )
