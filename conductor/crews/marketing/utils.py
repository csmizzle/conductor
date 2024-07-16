from typing import Union
from conductor.reports.models import ReportStyle
from conductor.reports.prompts import create_report_prompt
from conductor.reports.pydantic_templates import bulleted
from conductor.reports.pydantic_templates import narrative
from conductor.crews.models import TaskRun
from crewai import Task
import requests
from requests.models import Response
from redis import Redis
from bs4 import BeautifulSoup
from conductor.utils import is_gibberish


def write_report_prompt(
    report_style: ReportStyle,
    key_questions: list[str] = None,
) -> Union[str, None]:
    if key_questions:
        if report_style == ReportStyle.BULLETED:
            return create_report_prompt(
                report_style=ReportStyle.BULLETED,
                report_template_generator=bulleted.KeyQuestionsBulletedReportTemplate(
                    key_questions=key_questions
                ),
            )
        if report_style == ReportStyle.NARRATIVE:
            return create_report_prompt(
                report_style=ReportStyle.NARRATIVE,
                report_template_generator=narrative.KeyQuestionsNarrativeReportTemplate(
                    key_questions=key_questions
                ),
            )
    else:
        if report_style == ReportStyle.BULLETED:
            return create_report_prompt(
                report_style=ReportStyle.BULLETED,
                report_template_generator=bulleted.BulletedReportTemplate(),
            )
        if report_style == ReportStyle.NARRATIVE:
            return create_report_prompt(
                report_style=ReportStyle.NARRATIVE,
                report_template_generator=narrative.NarrativeReportTemplate(),
            )


def task_to_task_run(task: Task) -> TaskRun:
    return TaskRun(
        agent_role=task.agent.role,
        description=task.description,
        result=task.output.raw_output,
    )


def send_request_with_cache(
    method: str, url: str, cache: Redis, headers=None, cookies=None, timeout: int = None
) -> str:
    # check if url in redis
    cached_content = cache.get(url)
    if cached_content:
        return cached_content.decode("utf-8")
    # if not in redis, send request
    else:
        response = requests.request(
            method=method, url=url, headers=headers, cookies=cookies, timeout=timeout
        )
        if response.ok:
            clean_content = clean_html(response)
            cache.set(url, clean_content)
            return clean_content
        else:
            return f"Error: Unable to fetch page content for {url}."


def oxylabs_request(
    method: str,
    oxylabs_username: str,
    oxylabs_password: str,
    oxylabs_country: str,
    oxylabs_port: str,
    headers=None,
    cookies=None,
    **kwargs,
) -> Response:
    """
    Run Oxylabs request
    """
    response = requests.request(
        method=method,
        proxies={
            "http": f"http://{oxylabs_username}:{oxylabs_password}@{oxylabs_country}.oxylabs.io:{oxylabs_port}",
            "https": f"https://{oxylabs_username}:{oxylabs_password}@{oxylabs_country}.oxylabs.io:{oxylabs_port}",
        },
        headers=headers,
        cookies=cookies,
        **kwargs,
    )
    return response


def send_request_proxy(
    method: str,
    url: str,
    oxylabs_username: str,
    oxylabs_password: str,
    headers=None,
    cookies=None,
    **kwargs,
) -> Response:
    """
    Send a request to the specified URL.
    """
    try:
        page = oxylabs_request(
            method=method,
            oxylabs_username=oxylabs_username,
            oxylabs_password=oxylabs_password,
            oxylabs_country="pr",
            oxylabs_port=7777,
            url=url,
            headers=headers,
            cookies=cookies,
            **kwargs,
        )
        return page
    except Exception as e:
        # send normal request if oxylabs request fails
        print("Oxylabs request failed, sending normal request")
        print(e)
        try:
            print("Trying normal request ...")
            page = requests.request(method=method, url=url, **kwargs)
            return page
        except Exception as e:
            print("Normal request failed, returning error message.")
            print(e)
            return "Error: Unable to fetch page content for {url}."


def clean_html(response: Response) -> str:
    parsed = BeautifulSoup(response.content, "html.parser", from_encoding="iso-8859-1")
    text = parsed.get_text()
    text = "\n".join([i for i in text.split("\n") if i.strip() != ""])
    text = " ".join([i for i in text.split(" ") if i.strip() != ""])
    # remove all the special characters using regex
    # text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return f"Link: {response.url} \n Content: {text}"


def clean_and_remove_gibberish(response: Response, threshold: float = 0.9) -> str:
    """Clean and detect gibberish before using downstream

    Args:
        response (Response): HTML Response
        threshold (float): Threshold for gibberish detection

    Returns:
        str: Error determination
    """
    # clean text
    cleaned_text = clean_html(response)
    # check if gibberish
    if is_gibberish(text=cleaned_text, gibberish_threshold=threshold):
        return f"Error from {response.url}: Text is gibberish, ignore and do not include in answer."
    else:
        return cleaned_text


def send_request_proxy_with_cache(
    url: str,
    method: str,
    cache: Redis,
    oxylabs_username: str,
    oxylabs_password: str,
    headers: dict,
    cookies: dict,
    timeout: int,
    **kwargs,
) -> str:
    cached_content = cache.get(url)
    if cached_content:
        return cached_content.decode("utf-8")
    else:
        content = send_request_proxy(
            url=url,
            method=method,
            oxylabs_username=oxylabs_username,
            oxylabs_password=oxylabs_password,
            headers=headers,
            cookies=cookies if cookies else {},
            timeout=timeout,
            **kwargs,
        )
        if isinstance(content, Response) and content.ok:
            clean_content = clean_html(content)
            cache.set(url, clean_content)
            return clean_content
        else:
            return content
