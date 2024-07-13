"""
Test callback for crewai
"""
from conductor.crews.callbacks import (
    send_webhook_to_thread_sync,
    send_task_output_to_thread,
)
from crewai.tasks.task_output import TaskOutput
from discord.file import File
from functools import partial
import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))

TEST_REPORT = os.path.join(BASEDIR, "data", "test_report.pdf")
TEST_THREAD = os.getenv("TEST_THREAD")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")


def test_send_webhooks_to_thread_sync():
    sent_message = send_webhook_to_thread_sync(
        webhook_url=DISCORD_WEBHOOK_URL,
        token=DISCORD_BOT_TOKEN,
        thread_id=TEST_THREAD,
        content="Hello for the Test Suite!",
        username="Test Team",
    )
    assert sent_message is True
    sent_another_message = send_webhook_to_thread_sync(
        webhook_url=DISCORD_WEBHOOK_URL,
        token=DISCORD_BOT_TOKEN,
        thread_id=TEST_THREAD,
        content="Another message for the Test Suite!",
        username="Test Team",
    )
    assert sent_another_message is True


def test_send_task_output_to_thread():
    test_task_output = TaskOutput(
        description="Test Task Output",
        raw_output="Test Task Output Raw",
        exported_output="None",
    )
    sent_messages = send_task_output_to_thread(
        webhook_url=DISCORD_WEBHOOK_URL,
        token=DISCORD_BOT_TOKEN,
        task_output=test_task_output,
        thread_id=TEST_THREAD,
    )
    assert len(sent_messages) == 1
    assert sent_messages[0][0] is True
    assert sent_messages[0][1] == "Test Task Output Raw"


def test_send_long_output_to_thread():
    test_task_output = TaskOutput(
        description="Test Task Output",
        raw_output="A" * 5000,
        exported_output="None",
    )
    sent_messages = send_task_output_to_thread(
        webhook_url=DISCORD_WEBHOOK_URL,
        token=DISCORD_BOT_TOKEN,
        task_output=test_task_output,
        thread_id=TEST_THREAD,
    )
    assert len(sent_messages) == 3
    assert sent_messages[0][0] is True
    assert sent_messages[0][1] == "A" * 2000
    assert sent_messages[1][0] is True
    assert sent_messages[1][1] == "A" * 2000
    assert sent_messages[2][0] is True
    assert sent_messages[2][1] == "A" * 1000


def test_send_file_to_thread() -> None:
    sent_message = send_webhook_to_thread_sync(
        webhook_url=DISCORD_WEBHOOK_URL,
        token=DISCORD_BOT_TOKEN,
        thread_id=TEST_THREAD,
        content="Hello for the Test Suite! With a file this time!",
        file=File(TEST_REPORT),
        username="Test Team",
    )
    assert sent_message is True


def test_send_no_file_to_thread() -> None:
    sent_message = send_webhook_to_thread_sync(
        webhook_url=DISCORD_WEBHOOK_URL,
        token=DISCORD_BOT_TOKEN,
        thread_id=TEST_THREAD,
        content="Hello for the Test Suite! With a file this time!",
        file=None,
        username="Test Team",
    )
    assert sent_message is True


def test_partial_task_to_thread() -> None:
    """
    This is how we use partial functions in conductor-server to send task outputs
    """
    test_task_output = TaskOutput(
        description="Test Task Output",
        raw_output="Test Task Output Raw",
        exported_output="None",
    )
    partial_send_task_output = partial(
        send_task_output_to_thread,
        webhook_url=DISCORD_WEBHOOK_URL,
        token=DISCORD_BOT_TOKEN,
        thread_id=TEST_THREAD,
    )
    sent_messages = partial_send_task_output(task_output=test_task_output)
    assert len(sent_messages) == 1
    assert sent_messages[0][0] is True
    assert sent_messages[0][1] == "Test Task Output Raw"
