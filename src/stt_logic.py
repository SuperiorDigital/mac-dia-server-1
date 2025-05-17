# Implements Speech-to-Text functionality using mlx_audio
import os
import io
import tempfile
import asyncio
import gc  # 新增：用于手动回收内存
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
    每次请求时加载模型，识别后立即释放内存。
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

    # ----------------------
    # 每次请求时加载模型
    # ----------------------
    model = Model.from_pretrained(model_name)

    temp_file = None
    try:
        if isinstance(audio_file, str):
            audio_path = audio_file
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            if isinstance(audio_file, bytes):
                temp_file.write(audio_file)
            else:
                audio_file.seek(0)
                temp_file.write(audio_file.read())
            temp_file.close()
            audio_path = temp_file.name

        options = {}
        if language:
            options["language"] = language
        if prompt:
            options["prompt"] = prompt
        if temperature != 0.0:
            options["temperature"] = temperature

        result = model.generate(audio=audio_path, **options)

        # ----------------------
        # 推理后立即释放模型内存
        # ----------------------
        del model  # 删除模型对象
        gc.collect()  # 强制回收内存

        response = {
            "text": result
        }
        return response

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise

    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file.name}: {e}")
