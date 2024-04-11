"""
Get all historical messages from a channel
"""
from conductor.models import InternalKnowledgeChat
from conductor.database.aws import upload_dict_to_s3
import discord
from discord.ext import commands
import os
import uuid
import json
import logging


logger = logging.getLogger("discord")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    logger.info(f"We have logged in as {bot.user}")


@bot.command()
async def collect(ctx, channel_id: int):
    collect_id = uuid.uuid4()
    messages = []
    channel = bot.get_channel(channel_id)
    job_id = str(collect_id) + "-" + str(channel_id)
    logger.info(f"Starting collect job {job_id}")
    async for message in channel.history(limit=None):
        messages.append(
            InternalKnowledgeChat(
                source="discord",
                id=str(message.id),
                message=message.content,
                author=message.author.name,
                created_at=str(message.created_at),
                channel=message.channel.name,
            )
        )
    logger.info(f"Collected {len(messages)} messages from channel {channel_id}")
    logger.info(
        f"Uploading to S3: {os.getenv('DISCORD_S3_BUCKET')} with collect ID: {job_id}"
    )
    upload_dict_to_s3(
        data=json.dumps([message.dict() for message in messages], indent=4),
        bucket=os.getenv("DISCORD_S3_BUCKET"),
        key=f"{job_id}.json",
    )
    logger.info(
        f"Uploaded to S3: {os.getenv('DISCORD_S3_BUCKET')} with collect ID: {job_id}"
    )


bot.run(os.getenv("DISCORD_TOKEN"))
