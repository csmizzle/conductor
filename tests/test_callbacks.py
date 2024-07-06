"""
Test callback for crewai
"""
from conductor.crews.callbacks import send_webhook_to_thread_sync
import os

TEST_THREAD = os.getenv("TEST_THREAD")


def test_send_webhooks_to_thread_sync():
    sent_message = send_webhook_to_thread_sync(
        thread_id=TEST_THREAD, content="Hello for the Test Suite!", username="Test Team"
    )
    assert sent_message is True
    sent_another_message = send_webhook_to_thread_sync(
        thread_id=TEST_THREAD,
        content="Another message for the Test Suite!",
        username="Test Team",
    )
    assert sent_another_message is True
