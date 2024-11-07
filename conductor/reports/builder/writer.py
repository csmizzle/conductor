"""
Section writer from a report outline
- Decompose section into questions and sub-questions
- Get answers for each question
- Write answers to sections
"""
import dspy
from conductor.flow.retriever import ElasticRMClient
from conductor.flow.rag import CitationRAG, CitedAnswerWithCredibility
from conductor.reports.builder import models
from conductor.reports.builder import signatures
import concurrent.futures


class SectionWriter(dspy.Module):
    """
    Write a section from a report outline
    """

    def __init__(
        self,
        elastic_retriever: ElasticRMClient,
    ) -> None:
        self.rag = CitationRAG(elastic_retriever)
        self.generate_section_questions = dspy.ReAct(
            signatures.SectionQuestion,
            tools=[elastic_retriever],
        )
        self.generate_section = dspy.ReAct(
            signatures.Section,
            tools=[elastic_retriever],
        )

    def forward(
        self,
        section: models.SectionOutline,
    ) -> models.SourcedSection:
        # generate questions from section outline
        questions = self.generate_section_questions(
            section_outline_title=section.section_title,
            section_outline_content=section.section_content,
        )
        # collect answers for each question in parallel
        answers: list[CitedAnswerWithCredibility] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for question in questions.questions:
                futures.append(executor.submit(self.rag, question=question))
            for future in concurrent.futures.as_completed(futures):
                answers.append(future.result())
        # write sections
        generated_section = self.generate_section(
            section_outline_title=section.section_title,
            section_outline_content=section.section_content,
            questions=questions.questions,
            answers=answers,
        )
        # # map question sourcing to paragraph sentences
        sourced_paragraphs = []
        for paragraph in generated_section.section.paragraphs:
            sourced_sentences = []
            for sentence in paragraph.sentences:
                for answer in answers:
                    if sentence.question == answer.question:
                        sourced_sentence = models.SentenceWithAnswer(
                            content=sentence.content,
                            question=sentence.question,
                            answer=answer,
                        )
                        sourced_sentences.append(sourced_sentence)
            sourced_paragraph = models.SourcedParagraph(sentences=sourced_sentences)
            sourced_paragraphs.append(sourced_paragraph)
        sourced_section = models.SourcedSection(
            title=section.section_title,
            paragraphs=sourced_paragraphs,
        )
        return sourced_section


class ReportWriter:
    """
    Write a report from an outline
    """

    def __init__(
        self,
        elastic_retriever: ElasticRMClient,
    ) -> None:
        self.elastic_retriever = elastic_retriever

    def write_section(
        self, section_outline: models.SectionOutline
    ) -> models.SourcedSection:
        """
        Write a section from a report outline
        """
        writer = SectionWriter(self.elastic_retriever)
        return writer(section=section_outline)

    def write(
        self,
        outline: models.ReportOutline,
    ) -> models.Report:
        # write sections in parallel
        sections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for section in outline.report_sections:
                futures.append(
                    executor.submit(self.write_section, section_outline=section)
                )
            for future in concurrent.futures.as_completed(futures):
                sections.append(future.result())
        report = models.Report(sections=sections)
        return report


def write_section(
    section: models.SectionOutline,
    elastic_retriever: ElasticRMClient,
) -> dspy.Prediction:
    """
    Write a section from a report outline
    """
    writer = SectionWriter(elastic_retriever)
    return writer(section=section)


def write_report(
    outline: models.ReportOutline,
    elastic_retriever: ElasticRMClient,
) -> dspy.Prediction:
    """
    Write a report from an outline
    """
    writer = ReportWriter(elastic_retriever)
    return writer.write(outline=outline)
