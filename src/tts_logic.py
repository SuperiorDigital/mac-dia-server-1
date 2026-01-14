# Placeholder for TTS logic using mlx-audio

import io
import time
import gc  # 新增：用于手动回收内存
from .models import TTSRequest # Use relative import
from mlx_audio.tts.generate import generate_audio

def generate_speech_from_text_sync(request: TTSRequest) -> tuple[io.BytesIO, str]:
    """
    每次请求时加载模型，合成后立即释放内存。
    Args:
        request: TTS request details.
    Returns:
        A tuple containing an in-memory audio buffer (BytesIO) and the content type string.
    """
    print(f"Generating speech for text: '{request.input[:30]}...' using voice '{request.voice}'")

    import tempfile
    import os

    temp_dir = tempfile.gettempdir()
    temp_file_prefix = os.path.join(temp_dir, "tts_temp")

    output_format = request.response_format if hasattr(request, 'response_format') else "mp3"

    # ----------------------
    # 每次请求时加载模型并生成音频
    # ----------------------
    # 注意：mlx_audio.tts.generate.generate_audio 内部会自动加载模型

    generate_audio(
        text=request.input,
        model="mlx-community/Dia-1.6B-fp16",
        file_prefix=temp_file_prefix,
        audio_format=output_format,
        voice="af_heart",
        verbose=True
    )
    # ----------------------
    # 生成完毕后，尝试释放内存
    # ----------------------
    gc.collect()  # 强制回收内存，防止模型常驻

    temp_file_name = f"{temp_file_prefix}_000.{output_format}"
    if not os.path.exists(temp_file_name):
        temp_file_name = f"{temp_file_prefix}.{output_format}"
        if not os.path.exists(temp_file_name):
            raise FileNotFoundError(f"Generated audio file not found: {temp_file_name}")

    with open(temp_file_name, "rb") as f:
        audio_content = f.read()
    audio_buffer = io.BytesIO(audio_content)

    content_type_map = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
    }
    content_type = content_type_map.get(output_format, "audio/mpeg")

    try:
        os.remove(temp_file_name)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {temp_file_name}: {e}")

    print(f"Audio generation complete. Returning {len(audio_content)} bytes as {content_type}")
    return audio_buffer, content_type
