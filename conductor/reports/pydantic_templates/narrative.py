from conductor.reports.models import (
    ReportTemplatePromptGenerator,
    SectionTemplate,
    ParagraphTemplate,
)


class NarrativeReportTemplate(ReportTemplatePromptGenerator):
    key_questions = None
    sections = [
        SectionTemplate(
            title="Overview",
            paragraphs=[
                ParagraphTemplate(
                    title="Background",
                    content_template="Background of the company in long form narrative",
                ),
                ParagraphTemplate(
                    title="Key Personnel",
                    content_template="All identified personnel in long form narrative",
                ),
                ParagraphTemplate(
                    title="Products/Services",
                    content_template="All identified products/services in long form narrative",
                ),
                ParagraphTemplate(
                    title="Pricing",
                    content_template="All identified pricing in long form narrative",
                ),
                ParagraphTemplate(
                    title="Recent Events",
                    content_template="All identified recent events in long form narrative",
                ),
            ],
        ),
        SectionTemplate(
            title="Market Analysis",
            paragraphs=[
                ParagraphTemplate(
                    title="Market",
                    content_template="Identified market(s) in long form narrative",
                ),
                ParagraphTemplate(
                    title="TAM/SAM/SOM",
                    content_template="All identified TAM/SAM/SOM in long form narrative",
                ),
            ],
        ),
        SectionTemplate(
            title="SWOT Analysis",
            paragraphs=[
                ParagraphTemplate(
                    title="Strengths",
                    content_template="All identified strengths in long form narrative",
                ),
                ParagraphTemplate(
                    title="Weaknesses",
                    content_template="All identified weaknesses in long form narrative",
                ),
                ParagraphTemplate(
                    title="Opportunities",
                    content_template="All identified opportunities in long form narrative",
                ),
                ParagraphTemplate(
                    title="Threats",
                    content_template="All identified threats in long form narrative",
                ),
            ],
        ),
        SectionTemplate(
            title="Competition",
            paragraphs=[
                ParagraphTemplate(
                    title="Competitors",
                    content_template="All identified competitors in long form narrative",
                ),
                ParagraphTemplate(
                    title="Competitive Advantage",
                    content_template="All identified competitive advantages in long form narrative",
                ),
                ParagraphTemplate(
                    title="Competitive Disadvantage",
                    content_template="All identified competitive disadvantages in long form narrative",
                ),
            ],
        ),
        SectionTemplate(
            title="Sources",
            paragraphs=[
                ParagraphTemplate(
                    title="Links",
                    content_template="Links to sources in bulleted list",
                ),
            ],
        ),
    ]


class KeyQuestionsNarrativeReportTemplate(ReportTemplatePromptGenerator):
    def __init__(self, key_questions: list[str]) -> None:
        super().__init__()
        self.key_questions = key_questions
        self.sections = [
            SectionTemplate(
                title="Key Questions",
                paragraphs=[
                    ParagraphTemplate(
                        title=key_question,
                        content_template="Answer the question in long form narrative",
                    )
                    for key_question in self.key_questions
                ],
            ),
            SectionTemplate(
                title="Overview",
                paragraphs=[
                    ParagraphTemplate(
                        title="Background",
                        content_template="Background of the company in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Key Personnel",
                        content_template="All identified personnel in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Products/Services",
                        content_template="All identified products/services in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Pricing",
                        content_template="All identified pricing in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Recent Events",
                        content_template="All identified recent events in long form narrative",
                    ),
                ],
            ),
            SectionTemplate(
                title="Market Analysis",
                paragraphs=[
                    ParagraphTemplate(
                        title="Market",
                        content_template="Identified market(s) in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="TAM/SAM/SOM",
                        content_template="All identified TAM/SAM/SOM in long form narrative",
                    ),
                ],
            ),
            SectionTemplate(
                title="SWOT Analysis",
                paragraphs=[
                    ParagraphTemplate(
                        title="Strengths",
                        content_template="All identified strengths in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Weaknesses",
                        content_template="All identified weaknesses in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Opportunities",
                        content_template="All identified opportunities in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Threats",
                        content_template="All identified threats in long form narrative",
                    ),
                ],
            ),
            SectionTemplate(
                title="Competition",
                paragraphs=[
                    ParagraphTemplate(
                        title="Competitors",
                        content_template="All identified competitors in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Competitive Advantage",
                        content_template="All identified competitive advantages in long form narrative",
                    ),
                    ParagraphTemplate(
                        title="Competitive Disadvantage",
                        content_template="All identified competitive disadvantages in long form narrative",
                    ),
                ],
            ),
            SectionTemplate(
                title="Sources",
                paragraphs=[
                    ParagraphTemplate(
                        title="Links",
                        content_template="Links to sources in bulleted list",
                    ),
                ],
            ),
        ]
