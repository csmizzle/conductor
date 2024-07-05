"""
Task callbacks
- Discord thread webhook
"""
from aiohttp import ClientSession
from discord import Webhook
import os
import asyncio
import discord

intents = discord.Intents.default()
intents.guilds = True  # Ensure GUILDS intent is enabled
client = discord.Client(intents=intents)


async def send_webhook_to_thread(thread_id: int, content: str, username: str) -> None:
    """
    Send task output to discord thread
    """
    await client.login(token=os.getenv("DISCORD_BOT_TOKEN"))
    thread = await client.fetch_channel(thread_id)
    async with ClientSession() as session:
        webhook = Webhook.from_url(
            url=os.getenv("DISCORD_WEBHOOK_URL"), session=session
        )
        await webhook.send(thread=thread, content=content, username=username)
    await session.close()


def send_webhook_to_thread_sync(thread_id: str, content: str, username: str) -> None:
    """
    Send task output to discord thread
    """
    asyncio.run(send_webhook_to_thread(thread_id, content, username))
