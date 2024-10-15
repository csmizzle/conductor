from conductor.audio.transform import text_to_speech_file, SpeechFileResponse
from elevenlabs import VoiceSettings
import os


def test_text_to_speech_file() -> None:
    speech_response = text_to_speech_file(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        text="This is a test sentence",
        voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        model_id=os.getenv("ELEVENLABS_MODEL_ID"),
        output_path="test.mp3",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    assert isinstance(speech_response, SpeechFileResponse)
    assert os.path.exists("test.mp3")
    os.remove("test.mp3")
