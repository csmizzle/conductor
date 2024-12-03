"""
Use Claude to generate an outline of the report based on the user's input
"""
import dspy
from conductor.flow.rag import WebDocumentRetriever
from conductor.reports.builder import signatures
from conductor.reports.builder import models
from conductor.summarize.summarize import generate_master_summary
from pydantic import InstanceOf
from functools import partial
import concurrent.futures


class SectionOutlineBuilder:
    """
    Build a section outline with real time access to information
    """

    def __init__(
        self,
        specification: str,
        section_title: str,
        perspective: str,
        rag: InstanceOf[WebDocumentRetriever],
    ) -> None:
        self.specification = specification
        self.section_title = section_title
        self.perspective = perspective
        self.generate_section_outline_search = dspy.ChainOfThought(
            signatures.SectionOutlineSearch
        )
        self.generate_section_outline = dspy.ChainOfThought(signatures.SectionOutline)
        self.rag = rag

    def build(self) -> signatures.SectionOutline:
        """
        Generate an outline for the section based on the specification
        """
        search = self.generate_section_outline_search(
            specification=self.specification,
            perspective=self.perspective,
            section_title=self.section_title,
        )
        documents = self.rag(question=search.search)
        summary = generate_master_summary(
            documents=documents,
            question=search.search,
        )
        section_outline = self.generate_section_outline(
            perspective=self.perspective,
            specification=self.specification,
            section_title=self.section_title,
            documents_summary=summary.summary,
        )
        return section_outline


def generate_section_outline(
    specification: str,
    section_title: str,
    perspective: str,
    rag: InstanceOf[WebDocumentRetriever],
) -> models.SectionOutline:
    """
    Generate an outline for a section based on the specification
    """
    section_builder = SectionOutlineBuilder(
        specification=specification,
        section_title=section_title,
        perspective=perspective,
        rag=rag,
    )
    generated_section_outline = section_builder.build()
    return generated_section_outline.section_outline


class OutlineBuilder:
    """
    Outline builder that only uses the Language Model's knowledge to generate an outline for a report
    """

    def __init__(
        self,
        section_titles: list[str],
        perspective: str,
        specification: str,
        rag: InstanceOf[WebDocumentRetriever],
    ) -> None:
        self.section_titles = section_titles
        self.perspective = perspective
        self.specification = specification
        self.rag = rag

    def build(self) -> list[signatures.SectionOutline]:
        """
        Generate an outline for the report based on the specification in parallel
        """
        outlines: list[signatures.SectionOutline] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for title in self.section_titles:
                # Use functools.partial to pass multiple arguments
                func = partial(
                    generate_section_outline,
                    specification=self.specification,
                    section_title=title,
                    perspective=self.perspective,
                    rag=self.rag,
                )
                futures.append(executor.submit(func))
            for future in concurrent.futures.as_completed(futures):
                outlines.append(future.result())
        return outlines


class OutlineRefiner(dspy.Module):
    """
    Refine an outline based on the conversations
    """

    def __init__(self) -> None:
        self.refine_outline = dspy.ChainOfThought(signatures.RefinedOutline)

    def forward(
        self,
        perspective: str,
        specification: str,
        draft_outline: models.ReportOutline,
    ) -> dspy.Prediction:
        """
        Refine the outline based on the conversations
        """
        # summarize the team conversations
        refined_outline = self.refine_outline(
            perspective=perspective,
            draft_outline=draft_outline,
            specification=specification,
        )
        return refined_outline


def build_outline(
    report_title: str,
    section_titles: list[str],
    perspective: str,
    specification: str,
    rag: InstanceOf[WebDocumentRetriever],
) -> models.ReportOutline:
    """
    Generate an outline for the report based on the specification
    """
    outline_builder = OutlineBuilder(
        section_titles=section_titles,
        perspective=perspective,
        specification=specification,
        rag=rag,
    )
    sections = outline_builder.build()
    # sort outlines by section title to match the order of input section titles
    sorted_outlines = sorted(
        sections, key=lambda x: section_titles.index(x.section_title)
    )
    return models.ReportOutline(
        report_title=report_title,
        report_sections=sorted_outlines,
    )


def build_refined_outline(
    perspective: str,
    specification: str,
    draft_outline: models.ReportOutline,
) -> dspy.Prediction:
    """
    Refine the outline based on the conversations
    """
    outline_refiner = OutlineRefiner()
    refined_outline = outline_refiner(
        perspective=perspective,
        specification=specification,
        draft_outline=draft_outline,
    )
    return refined_outline
