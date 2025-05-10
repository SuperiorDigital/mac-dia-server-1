# Implements Speech-to-Text functionality using mlx_audio
import os
import io
import tempfile
import asyncio
from typing import Optional, BinaryIO, Dict, Any, Union

from mlx_audio.stt.models.whisper import Model

def transcribe_audio_sync(
    audio_file: Union[str, BinaryIO, bytes],
    model_name: str = "mlx-community/whisper-large-v3-turbo",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Transcribe audio using mlx_audio Whisper model.

    Args:
        audio_file: File path, file-like object, or bytes containing audio data
        model_name: Name of the Whisper model to use
        language: Optional language code for transcription
        prompt: Optional prompt to guide transcription
        temperature: Sampling temperature (0.0 means deterministic)

    Returns:
        Dictionary with transcription result
    """
    print(f"Transcribing audio using model: {model_name}")

    # Load the Whisper model
    model = Model.from_pretrained(model_name)

    temp_file = None
    try:
        # Handle different audio_file types
        if isinstance(audio_file, str):
            # If audio_file is a file path
            audio_path = audio_file
        else:
            # If audio_file is bytes or file-like object, save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            if isinstance(audio_file, bytes):
                temp_file.write(audio_file)
            else:
                # Assuming it's a file-like object
                audio_file.seek(0)
                temp_file.write(audio_file.read())
            temp_file.close()
            audio_path = temp_file.name

        # Set options for transcription
        options = {}
        if language:
            options["language"] = language
        if prompt:
            options["prompt"] = prompt
        if temperature != 0.0:
            options["temperature"] = temperature

        # Generate transcription
        result = model.generate(audio=audio_path, **options)

        # Format response to match OpenAI's API
        response = {
            "text": result
        }

        return response

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

    finally:
        # Clean up temp file if created
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file.name}: {e}")
