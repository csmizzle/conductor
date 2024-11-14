"""
Edits will happen at the section level
A report editor agent that is responsible for edit the report based on the complied information.
"""
import dspy
from conductor.reports.builder import models
from conductor.reports.builder import signatures
from loguru import logger
import concurrent.futures


class SectionEditor(dspy.Module):
    """
    Edit a section based on the compiled information
    """

    def __init__(self) -> None:
        self.edit_section = dspy.ChainOfThought(signatures.SectionForReview)

    def forward(
        self,
        perspective: str,
        section: models.SourcedSection,
    ) -> dspy.Prediction:
        """
        Edit the section based on the compiled information
        """
        section = self.edit_section(
            perspective=perspective,
            section=section,
        )
        return section


class ReportEditor:
    """
    Edit a report based on the compiled information
    """

    def __init__(self, report: models.Report, perspective: str) -> None:
        self.report = report
        self.perspective = perspective
        self.section_editor = SectionEditor()
        self.edit_full_report = dspy.ChainOfThought(signatures.ReportForReview)

    def _edit_section(self, section: models.Section) -> list[str]:
        """
        Edit a section based on the compiled information
        """
        logger.info(f"Editing section: {section.title}")
        # get section content
        section_content = []
        for paragraph in section.paragraphs:
            for sentence in paragraph.sentences:
                section_content.append(sentence.content)
        edited_section = self.section_editor(
            perspective=self.perspective,
            section=section_content,
        ).edited_section
        logger.info(f"Edited section: {section.title}")
        return edited_section

    def edit_sections(
        self,
    ) -> list[list[str]]:
        """
        Edit sections concurrently using a thread pool
        """
        edited_sections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for section in self.report.sections:
                futures.append(executor.submit(self._edit_section, section))
            for future in concurrent.futures.as_completed(futures):
                edited_sections.append(future.result())
        return edited_sections

    def edit_report(self) -> list[list[str]]:
        """
        Edit the report based on the compiled information
        """
        edited_sections = self.edit_sections()
        logger.info("Editing entire report ...")
        edited_report = self.edit_full_report(
            perspective=self.perspective,
            sections=edited_sections,
        ).edited_report
        logger.info("Edited entire report")
        return edited_report


def edit_report(
    perspective: str,
    report: models.Report,
) -> list[list[str]]:
    """
    Edit the report based on the compiled information
    """
    report_editor = ReportEditor(
        perspective=perspective,
        report=report,
    )
    edited_report = report_editor.edit_report()
    # # map the edited sections to the report
    # for section_idx, section in enumerate(report.sections):
    #     for paragraph_idx, paragraph in enumerate(edited_report[section_idx]):
    #         edited_paragraph =
    return edited_report
