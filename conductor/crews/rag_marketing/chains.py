"""
A small set of chains to augment the outputs of the Crew
"""
from conductor.llms import openai_gpt_4o
from conductor.reports.models import ReportStyleV2, ReportTone, ReportPointOfView
from conductor.crews.rag_marketing.prompts import report_section_prompt, section_parser
from conductor.crews.models import TaskRun, CrewRun
from conductor.reports.models import SectionV2, ReportV2, ParsedReportV2
from conductor.chains import run_set_graph_chain, run_timeline_chain, Graph, Timeline
from tqdm import tqdm
from langsmith import traceable


section_writer_chain = report_section_prompt | openai_gpt_4o | section_parser


def style_to_prompt(style: ReportStyleV2) -> str:
    # create extended prompt for the report style based on report style
    if style == ReportStyleV2.BULLETED:
        style = "as bulleted lists, avoiding long paragraphs."
    if style == ReportStyleV2.NARRATIVE:
        style = "as long form narratives, avoiding bullet points and short sentences."
    if style == ReportStyleV2.MIXED:
        style = "as a mixture of long form narratives and bulleted lists when it makes sense."
    return style


def task_run_to_report_section(
    task_run: TaskRun,
    style: ReportStyleV2,
    tone: ReportTone,
    point_of_view: ReportPointOfView,
    min_sentences: int = 3,
    max_sentences: int = 5,
    previous_sections: list[str] = None,
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
    style = style_to_prompt(style)
    return section_writer_chain.invoke(
        dict(
            title=task_run.section_name,
            style=style,
            tone=tone.value,
            context=task_run.result,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            point_of_view=point_of_view.value,
            previous_sections=previous_sections,
        )
    )


@traceable
def crew_run_to_report(
    crew_run: CrewRun,
    title: str,
    description: str,
    style: ReportStyleV2,
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
    previous_sections = []
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
                previous_sections="\n".join(previous_sections),
            )
            # create previous section context
            previous_section = ""
            previous_section += section.title + "\n"
            for paragraph in section.paragraphs:
                previous_section += " ".join(paragraph.sentences) + "\n"
            # append previous section to list
            previous_sections.append(previous_section)
            # collect new section and raw section
            sections.append(section)
            raw_sections.append(task_run.result)
        else:
            print(f"Not parsing {task_run.name}")
    parsed_report = ParsedReportV2(
        title=title, description=description, sections=sections
    )
    return ReportV2(report=parsed_report, raw=raw_sections)


# graph extraction
def extract_graph_from_task(
    task: TaskRun,
) -> Graph:
    """
    Extracts a graph from a given task.

    Args:
        task (TaskRun): The task to extract the graph from.

    Returns:
        Graph: The extracted graph.
    """
    return run_set_graph_chain(task.result)


def extract_graph_from_crew_run(
    crew_run: CrewRun,
    sections_filter: list[str] = None,
) -> Graph:
    """
    Extracts a graph from a given crew run.

    Args:
        crew_run (CrewRun): The crew run to extract the graph from.
        sections_filter (list[str]): The sections to extract the graph from. If None, all sections are extracted.
    Returns:
        Graph: The extracted graph.
    """
    text = ""
    for task in crew_run.tasks:
        if sections_filter:
            if task.name in sections_filter:
                text += task.result + "\n"
    return run_set_graph_chain(text=text)


def extract_graph_from_report(
    report: ReportV2,
    sections_filter: list[str] = None,
) -> Graph:
    """
    Extracts a graph from a given report.

    Args:
        report (ReportV2): The report to extract the graph from.
        sections_filter (list[str]): The sections to extract the graph from. If None, all sections are extracted.
    Returns:
        Graph: The extracted graph.
    """
    text = ""
    for section in report.report.sections:
        if sections_filter:
            if section.title in sections_filter:
                for paragraph in section.paragraphs:
                    text += " ".join(paragraph.sentences) + "\n"
    return run_set_graph_chain(text=text)


# timeline extraction
def extract_timeline_from_task(
    task: TaskRun,
) -> Timeline:
    """
    Extracts a timeline from a given task.

    Args:
        task (TaskRun): The task to extract the timeline from.

    Returns:
        Timeline: The extracted timeline.
    """
    return run_timeline_chain(task.result)


def extract_timeline_from_crew_run(
    crew_run: CrewRun,
    sections_filter: list[str] = None,
) -> Timeline:
    """
    Extracts a timeline from a given crew run.

    Args:
        crew_run (CrewRun): The crew run to extract the timeline from.
        sections_filter (list[str]): The sections to extract the timeline from. If None, all sections are extracted.
    Returns:
        Timeline: The extracted timeline.
    """
    text = ""
    for task in crew_run.tasks:
        if sections_filter:
            if task.name in sections_filter:
                text += task.result + "\n"
    return run_timeline_chain(text=text)


def extract_timeline_from_report(
    report: ReportV2,
    sections_filter: list[str] = None,
) -> Timeline:
    """
    Extracts a timeline from a given report.

    Args:
        report (ReportV2): The report to extract the timeline from.
        sections_filter (list[str]): The sections to extract the timeline from. If None, all sections are extracted.
    Returns:
        Timeline: The extracted timeline.
    """
    text = ""
    for section in report.report.sections:
        if sections_filter:
            if section.title in sections_filter:
                for paragraph in section.paragraphs:
                    text += " ".join(paragraph.sentences) + "\n"
    return run_timeline_chain(text=text)
