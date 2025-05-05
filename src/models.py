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
