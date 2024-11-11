"""
Output reports to static files
- PDF
- Docx
"""
from typing import Dict
from conductor.reports.builder import models
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ListStyle, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListItem,
    ListFlowable,
)


class PDFReportBuilder:
    def __init__(self, report: models.Report, title: str) -> None:
        self.report = report
        self.title = title
        self.source_map = self.build_source_map()
        self.bullet_style = ListStyle(name="BulletStyle", bulletType="bullet")

    def build_source_map(self) -> Dict[str, int]:
        """
        Build a source map for all the sourced sentences so that there are no repeats in the document
        """
        index = 1
        source_map = {}
        for section in self.report.sections:
            for paragraph in section.paragraphs:
                for sentence in paragraph.sentences:
                    for citation in sentence.answer.citations:
                        if citation not in source_map:
                            source_map[citation] = index
                            index += 1
        return source_map

    def build(self, filename: str) -> SimpleDocTemplate:
        """Generate a PDF report from the report

        Args:
            filename (str): Name of the report
            output_path (str): Path to save the report

        Returns:
            SimpleDocTemplate: PDF report
        """
        document_elements = []
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        title = Paragraph(self.title, styles["Title"])
        # append title to document elements
        document_elements.append(title)
        # Create a custom paragraph style with justified alignment
        justified_paragraph_style = ParagraphStyle(
            name="Justified",
            parent=styles["Normal"],  # Use 'Normal' as the base
            alignment=TA_JUSTIFY,  # Justified alignment
        )
        # build report sections
        for section in self.report.sections:
            section_title = Paragraph(section.title, styles["Heading1"])
            document_elements.append(section_title)
            for paragraph in section.paragraphs:
                sentences = []
                for sentence in paragraph.sentences:
                    sentence_content = sentence.content
                    if sentence.answer:
                        citation_references = sorted(
                            [
                                self.source_map[citation]
                                for citation in sentence.answer.citations
                            ]
                        )
                        superscript = "<sup>,</sup>".join(
                            [
                                f"<a href={sentence.answer.citations[index]}><sup>{citation}</sup></a>"
                                for index, citation in enumerate(citation_references)
                            ]
                        )
                        sentence_content += f"{superscript}"
                    sentences.append(sentence_content)
                paragraph_content = " ".join(sentences)
                paragraph_element = Paragraph(
                    paragraph_content, justified_paragraph_style
                )
                document_elements.append(paragraph_element)
                document_elements.append(Spacer(1, 12))
            # add space between paragraphs
            document_elements.append(Spacer(1, 12))
        # convert sources to a list
        sources_header = Paragraph("Sources", styles["Heading2"])
        document_elements.append(sources_header)
        list_flowable = []
        for source in self.source_map:
            list_flowable.append(
                ListItem(
                    Paragraph(f"<a href={source}>{source}</a>", styles["BodyText"]),
                    bulletText=self.source_map[source],
                )
            )
        sources_list = ListFlowable(
            list_flowable,
            style=self.bullet_style,
            bulletType="1",
            bulletFontSize=10,
            bulletIndent=0,
        )
        document_elements.append(sources_list)
        doc.build(document_elements)
        return doc
