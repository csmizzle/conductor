"""
A small set of chains to augment the outputs of the Crew
"""
from conductor.llms import openai_gpt_4o
from conductor.reports.models import ReportStyle, ReportTone, ReportPointOfView
from conductor.crews.rag_marketing.prompts import report_section_prompt, section_parser
from conductor.crews.models import TaskRun, CrewRun
from conductor.reports.models import SectionV2, ReportV2, ParsedReportV2
from tqdm import tqdm
from langsmith import traceable


section_writer_chain = report_section_prompt | openai_gpt_4o | section_parser


def task_run_to_report_section(
    task_run: TaskRun,
    style: ReportStyle,
    tone: ReportTone,
    point_of_view: ReportPointOfView,
    title: str = None,
    min_sentences: int = 3,
    max_sentences: int = 5,
) -> SectionV2:
    """
    Converts a TaskRun object into a ReportSection object.

    Args:
        task_run (TaskRun): The TaskRun object to convert.
        title (str): The title of the report section.
        style (ReportStyle): The style of the report section.
        tone (ReportTone): The tone of the report section.
        min_sentences (int): The minimum number of sentences for each paragraph.
        max_sentences (int): The maximum number of sentences for each paragraph.

    Returns:
        SectionV2: The converted Section Object.
    """
    return section_writer_chain.invoke(
        dict(
            title=title if title else "",
            style=style.value,
            tone=tone.value,
            context=task_run.result,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            point_of_view=point_of_view.value,
        )
    )


@traceable
def crew_run_to_report(
    crew_run: CrewRun,
    title: str,
    description: str,
    style: ReportStyle,
    tone: ReportTone,
    point_of_view: ReportPointOfView,
    min_sentences: int = 3,
    max_sentences: int = 5,
    section_titles_filter: list[str] = None,
    section_titles_endswith_filter: str = None,
) -> ReportV2:
    """
    Converts a CrewRun object into a Report object.

    Args:
        crew_run (CrewRun): The CrewRun object to convert.
        title (str): The title of the report.
        description (str): The description of the report.
        style (ReportStyle): The style of the report.
        tone (ReportTone): The tone of the report.
        point_of_view (ReportPointOfView): The point of view of the report.
        min_sentences (int): The minimum number of sentences for each paragraph.
        max_sentences (int): The maximum number of sentences for each paragraph.

    Returns:
        ReportV2: The converted Report Object.
    """
    if not section_titles_filter:
        section_titles_filter = []
    # set to star as default, hopefully this is a very rare occasion that a task name ends with *
    if not section_titles_endswith_filter:
        section_titles_endswith_filter = "*"
    raw_sections = []
    sections = []
    for task_run in tqdm(crew_run.tasks):
        # filter tasks based on task name and task name filter
        if not task_run.name.endswith(section_titles_endswith_filter):
            print(f"Parsing {task_run.name}")
            section = task_run_to_report_section(
                task_run=task_run,
                style=style,
                tone=tone,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                point_of_view=point_of_view,
            )
            sections.append(section)
            raw_sections.append(task_run.result)
        else:
            print(f"Not parsing {task_run.name}")
    parsed_report = ParsedReportV2(
        title=title, description=description, sections=sections
    )
    return ReportV2(report=parsed_report, raw=raw_sections)
