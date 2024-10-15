"""
Audio brief for a body of text using ElevenLabs
"""
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydantic import BaseModel
from typing import Iterator


class SpeechFileResponse(BaseModel):
    """
    Response from the ElevenLabs API
    """

    file_path: str
    response: Iterator[bytes]

    class Config:
        arbitrary_types_allowed = True


def text_to_speech_file(
    api_key: str,
    text: str,
    voice_id: str,
    model_id: str,
    output_path: str,
    voice_settings: VoiceSettings,
    output_format: str = "mp3_22050_32",
) -> SpeechFileResponse:
    client = ElevenLabs(api_key=api_key)
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        output_format=output_format,
        model_id=model_id,
        text=text,
        voice_settings=voice_settings,
    )
    with open(output_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    return SpeechFileResponse(
        file_path=output_path,
        response=response,
    )
