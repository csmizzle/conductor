"""
Task callbacks
- Discord thread webhook
"""
from aiohttp import ClientSession
from discord import Webhook
import os
import asyncio
import discord
from crewai.task import TaskOutput

intents = discord.Intents.default()
intents.guilds = True  # Ensure GUILDS intent is enabled
client = discord.Client(intents=intents)


async def send_webhook_to_thread(thread_id: int, content: str, username: str) -> bool:
    """
    Send task output to discord thread
    """
    await client.login(token=os.getenv("DISCORD_BOT_TOKEN"))
    thread = await client.fetch_channel(thread_id)
    session = ClientSession()
    webhook = Webhook.from_url(url=os.getenv("DISCORD_WEBHOOK_URL"), session=session)
    await webhook.send(thread=thread, content=content, username=username)
    await session.close()
    return True


def send_webhook_to_thread_sync(thread_id: str, content: str, username: str) -> None:
    """
    Send task output to discord thread
    """
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    sent_webhook = loop.run_until_complete(
        send_webhook_to_thread(thread_id, content, username)
    )
    return sent_webhook


def send_task_output_to_thread(
    task_output: TaskOutput, thread_id: str
) -> list[tuple[bool, str]]:
    """
    Send task output to discord thread
    """
    # break the content into 2000 character chunks and send them
    sent_content_chunks = []
    for i in range(0, len(task_output.raw_output), 2000):
        content_chunk = task_output.raw_output[i : i + 2000]
        sent_message = send_webhook_to_thread_sync(
            thread_id=thread_id,
            content=content_chunk,
            username="Marketing Team",
        )
        sent_content_chunks.append((sent_message, content_chunk))
    return sent_content_chunks
