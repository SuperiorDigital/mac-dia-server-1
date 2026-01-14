# Pydantic models for API requests and responses

from pydantic import BaseModel, Field
from typing import Literal, Optional

# Define the TTSRequest model according to OpenAI API specs.
class TTSRequest(BaseModel):
    model: str = Field(..., description="The TTS model to use, e.g., 'tts-1', 'tts-1-hd'.")
    input: str = Field(..., description="The text to synthesize.")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = Field(
        ..., description="The voice to use for synthesis."
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac"]] = Field(
        default="mp3", description="The format of the audio output."
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the speech, from 0.25 to 4.0."
    )

# Define the STTRequest model according to OpenAI API specs
class STTRequest(BaseModel):
    model: str = Field(default="whisper-large-v3", description="The STT model to use.")
    language: Optional[str] = Field(
        default=None, description="The language of the input audio. If not specified, the model will auto-detect."
    )
    prompt: Optional[str] = Field(
        default=None, description="Optional text to guide the model's style or continue a previous audio segment."
    )
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Field(
        default="json", description="The format of the transcript output."
    )
    temperature: Optional[float] = Field(
        default=0.0, description="The sampling temperature, between 0 and 1."
    )


# Model for internal use with voice cloning (not directly a request body since we use multipart form)
class VoiceCloneParams(BaseModel):
    """Parameters for voice cloning TTS generation."""
    input: str = Field(..., description="The text to synthesize.")
    ref_audio_path: str = Field(..., description="Path to the reference audio file for voice cloning.")
    ref_text: Optional[str] = Field(
        default=None, description="Transcript of the reference audio. If not provided, will be auto-transcribed."
    )
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav"]] = Field(
        default="mp3", description="The format of the audio output."
    )
    speed: Optional[float] = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the speech, from 0.25 to 4.0."
    )
