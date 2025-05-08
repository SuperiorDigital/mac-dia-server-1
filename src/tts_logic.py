# Placeholder for TTS logic using mlx-audio

import io
import time
from .models import TTSRequest # Use relative import
from mlx_audio.tts.generate import generate_audio

def generate_speech_from_text_sync(request: TTSRequest) -> tuple[io.BytesIO, str]:
    """Generate speech using mlx-audio and return audio data.

    Args:
        request: TTS request details.

    Returns:
        A tuple containing an in-memory audio buffer (BytesIO) and the content type string.
    """
    print(f"Generating speech for text: '{request.input[:30]}...' using voice '{request.voice}'")

    # Create temporary filename
    import tempfile
    import os

    temp_dir = tempfile.gettempdir()
    temp_file_prefix = os.path.join(temp_dir, "tts_temp")

    # Determine output format
    output_format = request.response_format if hasattr(request, 'response_format') else "mp3"

    # Call mlx-audio to generate audio file
    generate_audio(
        text=request.input,
        model_path="mlx-community/Dia-1.6B-4bit",  # Use specified model
        file_prefix=temp_file_prefix,
        audio_format=output_format,
        verbose=True  # Reduce output
    )

    # Determine generated filename
    # Note: generate_audio will create files in the format temp_file_prefix_000.{format}
    temp_file_name = f"{temp_file_prefix}_000.{output_format}"

    # Check if file exists
    if not os.path.exists(temp_file_name):
        # Try filename without index
        temp_file_name = f"{temp_file_prefix}.{output_format}"
        if not os.path.exists(temp_file_name):
            raise FileNotFoundError(f"Generated audio file not found: {temp_file_name}")

    # Read the generated audio file
    with open(temp_file_name, "rb") as f:
        audio_content = f.read()

    # Create memory buffer
    audio_buffer = io.BytesIO(audio_content)

    # Determine content type
    content_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
    }
    content_type = content_type_map.get(output_format, "audio/mpeg")  # Default to mp3

    # Delete temporary file
    try:
        os.remove(temp_file_name)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {temp_file_name}: {e}")

    print(f"Audio generation complete. Returning {len(audio_content)} bytes as {content_type}")
    return audio_buffer, content_type
