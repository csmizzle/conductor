"""
Audio brief for a body of text using ElevenLabs
"""
from conductor.reports.models import ReportV2
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


class ReportAudioTransformer:
    """
    Generate an audio briefing from a Evrim report
    """

    def __init__(
        self,
        report: ReportV2,
        api_key: str,
        voice_id: str,
        model_id: str,
        output_format: str = "mp3_22050_32",
    ) -> None:
        self.report = report
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format

    def transform_paragraph(
        self,
        section: int,
        paragraph: int,
        output_path: str,
        stability: float = 0.0,
        similarity_boost: float = 1.0,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> SpeechFileResponse:
        """Generate an audio speech file for a paragraph

        Args:
            section (int): section index
            paragraph (int): paragraph index
            output_path (str): output file path
            stability (float, optional): stability. Defaults to 0.0.
            similarity_boost (float, optional): similarity boost. Defaults to 1.0.
            style (float, optional): style. Defaults to 0.0.
            use_speaker_boost (bool, optional): use speaker boost. Defaults to True.

        Returns:
            SpeechFileResponse: file with response
        """
        paragraph = self.report.report.sections[section].paragraphs[paragraph]
        return text_to_speech_file(
            api_key=self.api_key,
            text=". ".join(paragraph.sentences),
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_path=output_path,
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost,
            ),
            output_format=self.output_format,
        )

    def transform_section(
        self,
        section: int,
        output_path: str,
        stability: float = 0.0,
        similarity_boost: float = 1.0,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> SpeechFileResponse:
        """Generate an audio speech file for a section

        Args:
            section (int): section index
            output_path (str): output file path
            stability (float, optional): stability of voice. Defaults to 0.0.
            similarity_boost (float, optional): deviate from original voice. Defaults to 1.0.
            style (float, optional): style drift. Defaults to 0.0.
            use_speaker_boost (bool, optional): use Elevenlabs speaker boost. Defaults to True.

        Returns:
            SpeechFileResponse: _description_
        """
        text = ""
        section = self.report.report.sections[section]
        for paragraph in section.paragraphs:
            text += ". ".join(paragraph.sentences)
        return text_to_speech_file(
            api_key=self.api_key,
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_path=output_path,
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost,
            ),
            output_format=self.output_format,
        )

    def transform_report(
        self,
        output_path: str,
        stability: float = 0.0,
        similarity_boost: float = 1.0,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> SpeechFileResponse:
        """Generate an audio speech file for the entire report

        Args:
            output_path (str): output file path
            stability (float, optional): stability of voice. Defaults to 0.0.
            similarity_boost (float, optional): deviate from original voice. Defaults to 1.0.
            style (float, optional): style drift. Defaults to 0.0.
            use_speaker_boost (bool, optional): use Elevenlabs speaker boost. Defaults to True.

        Returns:
            SpeechFileResponse: _description_
        """
        text = ""
        for section in self.report.report.sections:
            for paragraph in section.paragraphs:
                text += ". ".join(paragraph.sentences)
        return text_to_speech_file(
            api_key=self.api_key,
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_path=output_path,
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost,
            ),
            output_format=self.output_format,
        )
