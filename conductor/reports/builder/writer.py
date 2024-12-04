"""
Section writer from a report outline
- Decompose section into questions and sub-questions
- Get answers for each question
- Write answers to sections
"""
import dspy
from conductor.flow.rag import CitedAnswerWithCredibility
from conductor.reports.builder import models
from conductor.reports.builder import signatures
import concurrent.futures
from loguru import logger
from pydantic import InstanceOf


class SectionWriter(dspy.Module):
    """
    Write a section from a report outline
    """

    def __init__(
        self,
        rag: InstanceOf[dspy.Module],
    ) -> None:
        self.rag = rag
        self.generate_section_questions = dspy.ChainOfThought(
            signatures.SectionQuestion,
        )
        self.generate_section = dspy.ChainOfThought(
            signatures.Section,
        )

    def forward(
        self,
        section: models.SectionOutline,
        specification: str,
        perspective: str,
    ) -> models.SourcedSection:
        # generate questions from section outline
        logger.info(f"Generating questions for section: {section.section_title}")
        questions = self.generate_section_questions(
            section_outline_title=section.section_title,
            section_outline_content=section.section_content,
            specification=specification,
            perspective=perspective,
        )
        # collect answers for each question in parallel
        answers: list[CitedAnswerWithCredibility] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for question in questions.questions:
                futures.append(executor.submit(self.rag, question=question))
            for future in concurrent.futures.as_completed(futures):
                answers.extend(future.result())
        # slim answers by removing unnecessary fields like the documents
        logger.info("Slimming answers ...")
        if len(answers) > 0:
            answers = [answer.slim() for answer in answers]
        logger.info("Writing section ...")
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
            if paragraph:
                if isinstance(paragraph, models.Paragraph):
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
                    sourced_paragraph = models.SourcedParagraph(
                        sentences=sourced_sentences
                    )
                    sourced_paragraphs.append(sourced_paragraph)
        logger.info(f"Generated section: {section.section_title}")
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
        rag: InstanceOf[dspy.Module],
    ) -> None:
        self.rag = rag

    def write_section(
        self,
        section_outline: models.SectionOutline,
        specification: str,
        perspective: str,
    ) -> models.SourcedSection:
        """
        Write a section from a report outline
        """
        writer = SectionWriter(rag=self.rag)
        return writer(
            section=section_outline,
            specification=specification,
            perspective=perspective,
        )

    def write(
        self,
        outline: models.ReportOutline,
        specification: str,
        perspective: str,
    ) -> models.Report:
        # write sections in parallel
        sections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for section in outline.report_sections:
                futures.append(
                    executor.submit(
                        self.write_section,
                        section_outline=section,
                        specification=specification,
                        perspective=perspective,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                sections.append(future.result())
        report = models.Report(sections=sections)
        return report


def write_section(
    section: models.SectionOutline,
    specification: str,
    perspective: str,
    rag: InstanceOf[dspy.Module],
) -> dspy.Prediction:
    """
    Write a section from a report outline
    """
    writer = SectionWriter(rag=rag)
    return writer(
        section=section,
        specification=specification,
        perspective=perspective,
    )


def write_report(
    outline: models.ReportOutline,
    specification: str,
    perspective: str,
    rag: InstanceOf[dspy.Module],
) -> dspy.Prediction:
    """
    Write a report from an outline
    """
    writer = ReportWriter(rag=rag)
    return writer.write(
        outline=outline,
        specification=specification,
        perspective=perspective,
    )
