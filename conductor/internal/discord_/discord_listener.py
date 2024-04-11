"""
Ingest internal discord chats to create a knowledge base for the team.
"""
import discord
import os
from conductor.models import InternalKnowledgeChat
from pprint import pprint

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_message(message) -> None:
    pprint(
        InternalKnowledgeChat(
            source="discord",
            id=str(message.id),
            message=message.content,
            author=message.author.name,
            created_at=str(message.created_at),
            channel=message.channel.name,
        )
    )


client.run(os.getenv("DISCORD_TOKEN"))
