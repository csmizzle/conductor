from conductor.crews.marketing.utils import (
    create_report_prompt,
    oxylabs_request,
    write_report_prompt,
)
from conductor.reports.models import ReportStyle
from conductor.reports.pydantic_templates.bulleted import (
    BulletedReportTemplate,
    KeyQuestionsBulletedReportTemplate,
)
from conductor.reports.pydantic_templates.narrative import (
    NarrativeReportTemplate,
    KeyQuestionsNarrativeReportTemplate,
)
import os
from pydantic.v1 import ValidationError
import pytest


def test_bulleted_create_report_prompt_no_key_questions() -> None:
    """
    Test the create_report_prompt function.
    """
    report_style = ReportStyle.BULLETED
    report_template_generator = BulletedReportTemplate()
    result = create_report_prompt(
        report_style=report_style, report_template_generator=report_template_generator
    )
    assert isinstance(result, str)


def test_bulleted_create_report_prompt_key_questions() -> None:
    """
    Test the create_report_prompt function.
    """
    report_style = ReportStyle.BULLETED
    key_questions = ["What is the company name?", "What is the company website?"]
    report_template_generator = KeyQuestionsBulletedReportTemplate(
        key_questions=key_questions
    )
    result = create_report_prompt(
        report_style=report_style, report_template_generator=report_template_generator
    )
    assert isinstance(result, str)


def test_create_narrative_report_prompt_no_key_questions() -> None:
    """
    Test the create_report_prompt function.
    """
    report_style = ReportStyle.NARRATIVE
    report_template_generator = NarrativeReportTemplate()
    result = create_report_prompt(
        report_style=report_style, report_template_generator=report_template_generator
    )
    assert isinstance(result, str)


def test_create_narrative_report_prompt_key_questions() -> None:
    """
    Test the create_report_prompt function.
    """
    report_style = ReportStyle.BULLETED
    key_questions = ["What is the company name?", "What is the company website?"]
    report_template_generator = KeyQuestionsNarrativeReportTemplate(
        key_questions=key_questions
    )
    result = create_report_prompt(
        report_style=report_style, report_template_generator=report_template_generator
    )
    assert isinstance(result, str)


def test_bad_key_question_input_narrative() -> None:
    """
    Test the create_report_prompt function.
    """
    report_style = ReportStyle.NARRATIVE
    key_questions = "What is the company name?"
    report_template_generator = KeyQuestionsNarrativeReportTemplate(
        key_questions=key_questions
    )
    with pytest.raises(ValidationError):
        create_report_prompt(
            report_style=report_style,
            report_template_generator=report_template_generator,
        )


def test_write_bulleted_report_prompt_no_key_questions() -> None:
    """
    Test the write_report_prompt function.
    """
    report_style = ReportStyle.BULLETED
    result = write_report_prompt(report_style=report_style)
    assert isinstance(result, str)


def test_write_bulleted_report_prompt_key_questions() -> None:
    """
    Test the write_report_prompt function.
    """
    report_style = ReportStyle.BULLETED
    key_questions = ["What is the company name?", "What is the company website?"]
    result = write_report_prompt(report_style=report_style, key_questions=key_questions)
    assert isinstance(result, str)


def test_write_narrative_report_prompt_no_key_questions() -> None:
    """
    Test the write_report_prompt function.
    """
    report_style = ReportStyle.NARRATIVE
    result = write_report_prompt(report_style=report_style)
    assert isinstance(result, str)


def test_write_narrative_report_prompt_key_questions() -> None:
    """
    Test the write_report_prompt function.
    """
    report_style = ReportStyle.NARRATIVE
    key_questions = ["What is the company name?", "What is the company website?"]
    result = write_report_prompt(report_style=report_style, key_questions=key_questions)
    assert isinstance(result, str)


def test_write_narrative_report_prompt_key_question() -> None:
    """
    Test the write_report_prompt function.
    """
    report_style = ReportStyle.NARRATIVE
    key_questions = ["What is the company name?"]
    result = write_report_prompt(report_style=report_style, key_questions=key_questions)
    assert isinstance(result, str)


def test_write_narrative_report_prompt_wrong_key_questions() -> None:
    """
    Test the write_report_prompt function.
    """
    report_style = ReportStyle.NARRATIVE
    key_questions = "What is the company name?"
    with pytest.raises(ValidationError):
        write_report_prompt(report_style=report_style, key_questions=key_questions)


def test_oxylabs_request() -> None:
    oxylabs_username = os.getenv("OXYLABS_USERNAME")
    oxylabs_password = os.getenv("OXYLABS_PASSWORD")
    oxylabs_port = 7777
    oxylabs_response = oxylabs_request(
        method="GET",
        oxylabs_username=oxylabs_username,
        oxylabs_password=oxylabs_password,
        oxylabs_country="pr",
        oxylabs_port=oxylabs_port,
        url="https://ip.oxylabs.io/",
    )
    assert oxylabs_response.status_code == 200
