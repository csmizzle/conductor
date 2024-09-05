from conductor.reports.models import (
    SourcedHyDESectionTemplate,
    SourcedHyDEReportTemplate,
    ReportStyleV2,
    ReportTone,
    ReportPointOfView,
)


class BasicSourcedHyDEReportTemplateGenerator:
    """Generates a report template with sourced HyDE sections."""

    def __init__(
        self,
        title: str,
        style: ReportStyleV2,
        tone: ReportTone,
        point_of_view: ReportPointOfView,
        hyde_context: str,
        section_titles: list[str],
        hyde_section_objectives: list[str],
    ) -> None:
        if len(section_titles) != len(hyde_section_objectives):
            raise ValueError(
                "section_titles and hyde_section_objectives must be the same length"
            )
        self.title = title
        self.style = style
        self.tone = tone
        self.point_of_view = point_of_view
        self.hyde_context = hyde_context
        self.section_titles = section_titles
        self.hyde_section_objectives = hyde_section_objectives

    def generate(self) -> SourcedHyDEReportTemplate:
        # build section templates
        section_templates = []
        for title, objective in zip(self.section_titles, self.hyde_section_objectives):
            section_template = SourcedHyDESectionTemplate(
                title=title,
                style=self.style,
                tone=self.tone,
                point_of_view=self.point_of_view,
                hyde_context=self.hyde_context,
                hyde_objective=objective,
            )
            section_templates.append(section_template)
        # build report template
        report_template = SourcedHyDEReportTemplate(
            title=self.title,
            section_templates=section_templates,
        )
        return report_template
