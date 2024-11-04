"""
Use Claude to generate an outline of the report based on the user's input
"""
import dspy
from conductor.reports.builder import signatures
from functools import partial
import concurrent.futures


class OutlineBuilder(dspy.Module):
    """
    Outline builder that only uses the Language Model's knowledge to generate an outline for a report
    """

    def __init__(self, section_titles: list[str]):
        self.section_titles = section_titles
        self.generate_section_outline = dspy.ChainOfThought(signatures.SectionOutline)

    def forward(self, specification: str) -> list[signatures.SectionOutline]:
        """
        Generate an outline for the report based on the specification in parallel
        """
        outlines: list[signatures.SectionOutline] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for title in self.section_titles:
                # Use functools.partial to pass multiple arguments
                func = partial(
                    self.generate_section_outline,
                    specification=specification,
                    section_title=title,
                )
                futures.append(executor.submit(func))
            for future in concurrent.futures.as_completed(futures):
                outlines.append(future.result())
        return outlines


def build_outline(
    specification: str, section_titles: list[str]
) -> list[signatures.SectionOutline]:
    """
    Generate an outline for the report based on the specification
    """
    outline_builder = OutlineBuilder(section_titles=section_titles)
    outline = outline_builder(specification=specification)
    sections = [section.section_outline for section in outline]
    # sort outlines by section title to match the order of input section titles
    sorted_outlines = sorted(
        sections, key=lambda x: section_titles.index(x.section_title)
    )
    return sorted_outlines
