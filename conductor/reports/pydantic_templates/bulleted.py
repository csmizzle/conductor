from conductor.reports.models import (
    ReportTemplatePromptGenerator,
    SectionTemplate,
    ParagraphTemplate,
)


class BulletedReportTemplate(ReportTemplatePromptGenerator):
    key_questions = None
    sections = [
        SectionTemplate(
            title="Overview",
            paragraphs=[
                ParagraphTemplate(
                    title="Background",
                    content_template="Background of the company in bullet points",
                ),
                ParagraphTemplate(
                    title="Key Personnel",
                    content_template="All identified personnel in bullet points",
                ),
                ParagraphTemplate(
                    title="Products/Services",
                    content_template="All identified products/services in bullet points",
                ),
                ParagraphTemplate(
                    title="Pricing",
                    content_template="All identified pricing in bullet points",
                ),
                ParagraphTemplate(
                    title="Recent Events",
                    content_template="All identified recent events in bullet points",
                ),
            ],
        ),
        SectionTemplate(
            title="Market Analysis",
            paragraphs=[
                ParagraphTemplate(
                    title="Market",
                    content_template="Identified market(s) in bullet points",
                ),
                ParagraphTemplate(
                    title="TAM/SAM/SOM",
                    content_template="All identified TAM/SAM/SOM in bullet points",
                ),
            ],
        ),
        SectionTemplate(
            title="SWOT Analysis",
            paragraphs=[
                ParagraphTemplate(
                    title="Strengths",
                    content_template="All identified strengths in bullet points",
                ),
                ParagraphTemplate(
                    title="Weaknesses",
                    content_template="All identified weaknesses in bullet points",
                ),
                ParagraphTemplate(
                    title="Opportunities",
                    content_template="All identified opportunities in bullet points",
                ),
                ParagraphTemplate(
                    title="Threats",
                    content_template="All identified threats in bullet points",
                ),
            ],
        ),
        SectionTemplate(
            title="Competition",
            paragraphs=[
                ParagraphTemplate(
                    title="Competitors",
                    content_template="All identified competitors strengths, weaknesses, opportunities in bullet points",
                ),
                ParagraphTemplate(
                    title="Risk Analysis",
                    content_template="Risk analysis in bullet points",
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
                    content_template="Links to sources as bullet points",
                ),
            ],
        ),
    ]


class KeyQuestionsBulletedReportTemplate(ReportTemplatePromptGenerator):
    def __init__(self, key_questions: list[str]) -> None:
        super().__init__()
        self.key_questions = key_questions
        self.sections = [
            SectionTemplate(
                title="Key Questions",
                paragraphs=[
                    ParagraphTemplate(
                        title=key_question,
                        content_template="Answer to the key question in bullet points",
                    )
                    for key_question in self.key_questions
                ],
            ),
            SectionTemplate(
                title="Overview",
                paragraphs=[
                    ParagraphTemplate(
                        title="Background",
                        content_template="Background of the company in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Key Personnel",
                        content_template="All identified personnel in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Products/Services",
                        content_template="All identified products/services in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Pricing",
                        content_template="All identified pricing in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Recent Events",
                        content_template="All identified recent events in bullet points",
                    ),
                ],
            ),
            SectionTemplate(
                title="Market Analysis",
                paragraphs=[
                    ParagraphTemplate(
                        title="Market",
                        content_template="Identified market(s) in bullet points",
                    ),
                    ParagraphTemplate(
                        title="TAM/SAM/SOM",
                        content_template="All identified TAM/SAM/SOM in bullet points",
                    ),
                ],
            ),
            SectionTemplate(
                title="SWOT Analysis",
                paragraphs=[
                    ParagraphTemplate(
                        title="Strengths",
                        content_template="All identified strengths in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Weaknesses",
                        content_template="All identified weaknesses in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Opportunities",
                        content_template="All identified opportunities in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Threats",
                        content_template="All identified threats in bullet points",
                    ),
                ],
            ),
            SectionTemplate(
                title="Competition",
                paragraphs=[
                    ParagraphTemplate(
                        title="Competitors",
                        content_template="All identified competitors strengths, weaknesses, opportunities and threats in bullet points",
                    ),
                    ParagraphTemplate(
                        title="Risk Analysis",
                        content_template="Risk analysis in bullet points",
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
                        content_template="Links to sources as bullet points",
                    ),
                ],
            ),
        ]
