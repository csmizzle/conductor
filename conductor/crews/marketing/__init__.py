from conductor.reports.outputs import string_to_report
from conductor.crews.marketing.crew import UrlMarketingCrew
from conductor.reports.models import Report, ReportStyle
from conductor.crews.models import CrewRun
from langsmith import traceable


@traceable
def run_marketing_crew(
    url: str,
    report_style: ReportStyle,
    output_log_file: bool | str = None,
    step_callback=None,
    task_callback=None,
) -> CrewRun:
    """
    Run a marketing crew on a URL
    """
    crew = UrlMarketingCrew(
        url=url,
        report_style=report_style,
        output_log_file=output_log_file,
        step_callback=step_callback,
        task_callback=task_callback,
    )
    crew_run = crew.run()
    return crew_run


def create_marketing_report(
    crew_run: CrewRun,
    report_style: ReportStyle,
) -> Report:
    """
    Run a marketing report on a URL
    """
    try:
        parsed_report = string_to_report(
            string=crew_run.result,
        )
        report = Report(
            report=parsed_report,
            style=report_style,
            raw=crew_run.result,
        )
    except Exception as e:
        print("Error parsing report. Returning report with raw output.")
        print(e)
        report = Report(
            report=None,
            style=report_style,
            raw=crew_run.result,
        )
    return report
