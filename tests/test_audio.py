from conductor.audio.transform import (
    text_to_speech_file,
    SpeechFileResponse,
    ReportAudioTransformer,
)
from conductor.reports.models import ReportV2
from tests.constants import REPORT_V2_JSON
from elevenlabs import VoiceSettings
import os

test_report = ReportV2.model_validate(REPORT_V2_JSON)


def test_text_to_speech_file() -> None:
    speech_response = text_to_speech_file(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        text="This is a test sentence",
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model_id=os.getenv("ELEVENLABS_MODEL_ID"),
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    assert isinstance(speech_response, SpeechFileResponse)
    speech_response.save("test.mp3")
    assert os.path.exists("test.mp3")
    os.remove("test.mp3")


def test_paragraph_to_speech_file() -> None:
    transformer = ReportAudioTransformer(
        report=test_report,
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model_id=os.getenv("ELEVENLABS_MODEL_ID"),
    )
    paragraph_transformed = transformer.transform_paragraph(
        section=0,
        paragraph=0,
    )
    assert isinstance(paragraph_transformed, SpeechFileResponse)
    paragraph_transformed.save("test_paragraph.mp3")
    assert os.path.exists("test_paragraph.mp3")
    os.remove("test_paragraph.mp3")


def test_section_to_speech_file() -> None:
    transformer = ReportAudioTransformer(
        report=test_report,
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model_id=os.getenv("ELEVENLABS_MODEL_ID"),
    )
    section_transformed = transformer.transform_section(
        section=0,
    )
    assert isinstance(section_transformed, SpeechFileResponse)
    section_transformed.save("test_section.mp3")
    assert os.path.exists("test_section.mp3")
    os.remove("test_section.mp3")


def test_report_to_speech_file() -> None:
    transformer = ReportAudioTransformer(
        report=test_report,
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model_id=os.getenv("ELEVENLABS_MODEL_ID"),
    )
    report_transformed = transformer.transform_report()
    assert isinstance(report_transformed, SpeechFileResponse)
    report_transformed.save("test_report.mp3")
    assert os.path.exists("test_report.mp3")
    os.remove("test_report.mp3")
