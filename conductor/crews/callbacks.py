"""
Task callbacks
- Discord thread webhook
"""
from aiohttp import ClientSession
from discord import Webhook
from discord.utils import MISSING
from typing import Any
from discord.file import File
from requests.models import Response
import discord
from crewai.task import TaskOutput
from discord_webhook import DiscordWebhook

intents = discord.Intents.default()
intents.guilds = True  # Ensure GUILDS intent is enabled
client = discord.Client(intents=intents)

# set MISSING for file if not provided
MISSING: Any = MISSING


async def send_webhook_to_thread(
    token: str,
    webhook_url: str,
    thread_id: int,
    content: str,
    username: str,
    file: File = None,
) -> bool:
    """
    Send task output to discord thread

    Args:
        token (str): The token used for authentication.
        webhook_url (str): The URL of the webhook.
        thread_id (int): The ID of the thread to send the webhook to.
        content (str): The content of the webhook message.
        username (str): The username to display for the webhook message.
        file (File, optional): The file to attach to the webhook message. Defaults to None.

    Returns:
        bool: True if the webhook was sent successfully, False otherwise.
    """
    await client.login(token=token)
    thread = await client.fetch_channel(thread_id)
    async with ClientSession() as session:
        webhook = Webhook.from_url(url=webhook_url, session=session)
        # omit file if not provided
        await webhook.send(
            thread=thread,
            content=content,
            username=username,
            file=file if file else MISSING,
        )
    return True


def send_webhook_to_thread_sync(
    token: str,
    webhook_url: str,
    thread_id: str,
    content: str,
    username: str,
    file: str = None,
    filename: str = None,
) -> Response | DiscordWebhook:
    """
    Send task output to discord thread synchronously.

    Args:
        token (str): The authentication token.
        webhook_url (str): The URL of the webhook.
        thread_id (str): The ID of the thread.
        content (str): The content to send.
        username (str): The username to display.
        file (File, optional): The file to attach (default: None).

    Returns:
        None: This function does not return any value.
    """
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    # if loop.is_closed():
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    # sent_webhook = asyncio.run(
    #     send_webhook_to_thread(
    #         token=token,
    #         webhook_url=webhook_url,
    #         thread_id=thread_id,
    #         content=content,
    #         username=username,
    #         file=file,
    #     )
    # )
    if not file:
        webhook = DiscordWebhook(
            url=webhook_url, content=content, username=username, thread_id=thread_id
        ).execute()
    else:
        with open(file, "rb") as file_:
            webhook = DiscordWebhook(
                url=webhook_url,
                content=content,
                username=username,
                thread_id=thread_id,
            )
            webhook.add_file(file=file_, filename=filename)
            webhook.execute()
    return webhook


def send_task_output_to_thread(
    token: str,
    webhook_url: str,
    thread_id: str,
    task_output: TaskOutput,
) -> list[tuple[bool, str]]:
    """
    Send task output to discord thread.

    Args:
        token (str): The authentication token for the Discord API.
        webhook_url (str): The URL of the webhook to send the message to.
        thread_id (str): The ID of the thread to send the message to.
        task_output (TaskOutput): The output of the task.

    Returns:
        list[tuple[bool, str]]: A list of tuples containing the status of each sent message and the content of each chunk.

    """
    # break the content into 2000 character chunks and send them
    sent_content_chunks = []
    for i in range(0, len(task_output.raw), 2000):
        content_chunk = task_output.raw[i : i + 2000]
        sent_message = send_webhook_to_thread_sync(
            token=token,
            webhook_url=webhook_url,
            thread_id=thread_id,
            content=content_chunk,
            username="Marketing Team",
        )
        sent_content_chunks.append((sent_message, content_chunk))
    return sent_content_chunks
