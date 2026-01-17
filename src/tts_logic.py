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


def generate_cloned_speech_sync(
    text: str,
    ref_audio_path: str,
    ref_text: str = None,
    output_format: str = "mp3",
    speed: float = 1.0
) -> tuple[io.BytesIO, str]:
    """
    Generate speech using voice cloning from a reference audio.
    
    Args:
        text: The text to synthesize.
        ref_audio_path: Path to the reference audio file for voice cloning.
        ref_text: Optional transcript of the reference audio.
        output_format: Output audio format (mp3, wav, etc.)
        speed: Speech speed multiplier.
    
    Returns:
        A tuple containing an in-memory audio buffer (BytesIO) and the content type string.
    """
    print(f"Generating cloned speech for text: '{text[:50]}...' using reference audio: '{ref_audio_path}'")
    
    import tempfile
    import os
    import glob
    
    temp_dir = tempfile.gettempdir()
    temp_file_prefix = os.path.join(temp_dir, "tts_clone_temp")
    
    # Clean up any previous temp files first
    for old_file in glob.glob(f"{temp_file_prefix}*"):
        try:
            os.remove(old_file)
        except:
            pass
    
    print(f"Calling generate_audio with ref_audio={ref_audio_path}, ref_text={ref_text}")
    
    # Generate audio with voice cloning using CSM (Sesame's Conversational Speech Model)
    try:
        generate_audio(
            text=text,
            model="mlx-community/csm-1b",  # CSM (Sesame) for voice cloning - supported by mlx_audio and cached locally
            file_prefix=temp_file_prefix,
            audio_format=output_format,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            speed=speed,
            verbose=True
        )
    except Exception as e:
        print(f"Error in generate_audio: {e}")
        raise
    
    # Force garbage collection after generation
    gc.collect()
    
    # Find the generated file using glob (handles various naming patterns and formats)
    # F5-TTS may output different formats than requested
    all_formats = [output_format, "wav", "mp3", "flac"]
    possible_patterns = []
    for fmt in all_formats:
        possible_patterns.extend([
            f"{temp_file_prefix}_000.{fmt}",
            f"{temp_file_prefix}.{fmt}",
            f"{temp_file_prefix}_*.{fmt}",
            f"{temp_file_prefix}*.{fmt}",
        ])
    
    temp_file_name = None
    for pattern in possible_patterns:
        matches = glob.glob(pattern)
        if matches:
            temp_file_name = matches[0]
            print(f"Found output file: {temp_file_name}")
            break
    
    if not temp_file_name:
        # List ALL files in temp dir matching prefix
        all_temp_files = glob.glob(f"{temp_file_prefix}*")
        # Also list recent audio files in temp dir
        import time
        recent_audio = []
        for f in glob.glob(os.path.join(temp_dir, "*.mp3")) + glob.glob(os.path.join(temp_dir, "*.wav")):
            if os.path.getmtime(f) > time.time() - 60:  # Last 60 seconds
                recent_audio.append(f)
        print(f"Files matching prefix: {all_temp_files}")
        print(f"Recent audio files in temp: {recent_audio}")
        raise FileNotFoundError(f"Generated audio file not found. Searched patterns: {possible_patterns[:4]}")
    
    # Read the generated audio
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
    
    # Clean up temp file
    try:
        os.remove(temp_file_name)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {temp_file_name}: {e}")
    
    print(f"Voice cloning complete. Returning {len(audio_content)} bytes as {content_type}")
    return audio_buffer, content_type


def chunk_text(text: str, max_words: int = 300) -> list[str]:
    """
    Split text into chunks at sentence boundaries.
    
    Args:
        text: The full text to chunk.
        max_words: Maximum words per chunk (default 300 for TTS models).
    
    Returns:
        List of text chunks.
    """
    import re
    
    # Split into sentences (handles ., !, ?, and common abbreviations)
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_endings.split(text)
    
    # If no sentence boundaries found, fall back to simpler split
    if len(sentences) <= 1:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_words = len(sentence.split())
        
        # If single sentence exceeds max, split by words
        if sentence_words > max_words:
            # Flush current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Split long sentence by words
            words = sentence.split()
            for i in range(0, len(words), max_words):
                chunk_words = words[i:i + max_words]
                chunks.append(' '.join(chunk_words))
        elif current_word_count + sentence_words > max_words:
            # Current chunk is full, start new one
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def concatenate_audio_files(audio_files: list[str], output_path: str, output_format: str = "mp3"):
    """
    Concatenate multiple audio files into one.
    Uses pydub for audio concatenation.
    
    Args:
        audio_files: List of paths to audio files to concatenate.
        output_path: Path for the output concatenated file.
        output_format: Output format (mp3, wav, etc.)
    """
    from pydub import AudioSegment
    
    combined = AudioSegment.empty()
    
    for audio_file in audio_files:
        # Detect format from extension
        ext = audio_file.split('.')[-1].lower()
        segment = AudioSegment.from_file(audio_file, format=ext)
        combined += segment
    
    # Export with appropriate format
    combined.export(output_path, format=output_format)
    print(f"Concatenated {len(audio_files)} audio files into {output_path}")


def generate_cloned_speech_long_sync(
    text: str,
    ref_audio_path: str,
    ref_text: str = None,
    output_format: str = "mp3",
    speed: float = 1.0,
    max_words_per_chunk: int = 300,
    progress_callback = None
) -> tuple[io.BytesIO, str]:
    """
    Generate long-form speech using voice cloning.
    Chunks the text, generates audio for each chunk, and concatenates.
    
    Args:
        text: The full text to synthesize (can be 2000-20000+ words).
        ref_audio_path: Path to the reference audio file for voice cloning.
        ref_text: Optional transcript of the reference audio.
        output_format: Output audio format (mp3, wav, etc.)
        speed: Speech speed multiplier.
        max_words_per_chunk: Maximum words per chunk (default 300).
        progress_callback: Optional callback(current_chunk, total_chunks) for progress.
    
    Returns:
        A tuple containing an in-memory audio buffer (BytesIO) and the content type string.
    """
    import tempfile
    import os
    import glob
    
    word_count = len(text.split())
    print(f"Starting long-form voice cloning: {word_count} words")
    
    # Chunk the text
    chunks = chunk_text(text, max_words=max_words_per_chunk)
    print(f"Split into {len(chunks)} chunks")
    
    # If only one chunk, use regular generation
    if len(chunks) == 1:
        return generate_cloned_speech_sync(
            text=chunks[0],
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            output_format=output_format,
            speed=speed
        )
    
    temp_dir = tempfile.gettempdir()
    chunk_audio_files = []
    
    try:
        # Generate audio for each chunk
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
            
            if progress_callback:
                progress_callback(i + 1, len(chunks))
            
            chunk_prefix = os.path.join(temp_dir, f"tts_chunk_{i:04d}")
            
            # Clean up any previous temp files for this chunk
            for old_file in glob.glob(f"{chunk_prefix}*"):
                try:
                    os.remove(old_file)
                except:
                    pass
            
            # Generate audio for this chunk
            generate_audio(
                text=chunk,
                model="mlx-community/csm-1b",
                file_prefix=chunk_prefix,
                audio_format=output_format,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                speed=speed,
                verbose=False  # Less verbose for chunks
            )
            
            # Find the generated file
            all_formats = [output_format, "wav", "mp3", "flac"]
            chunk_file = None
            for fmt in all_formats:
                for pattern in [f"{chunk_prefix}_000.{fmt}", f"{chunk_prefix}.{fmt}"]:
                    if os.path.exists(pattern):
                        chunk_file = pattern
                        break
                    matches = glob.glob(pattern)
                    if matches:
                        chunk_file = matches[0]
                        break
                if chunk_file:
                    break
            
            if not chunk_file:
                raise FileNotFoundError(f"Failed to generate audio for chunk {i+1}")
            
            chunk_audio_files.append(chunk_file)
            print(f"Chunk {i+1} complete: {chunk_file}")
            
            # Garbage collect between chunks
            gc.collect()
        
        # Concatenate all chunks
        output_path = os.path.join(temp_dir, f"tts_final_output.{output_format}")
        concatenate_audio_files(chunk_audio_files, output_path, output_format)
        
        # Read the final audio
        with open(output_path, "rb") as f:
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
        
        print(f"Long-form voice cloning complete. {len(chunks)} chunks -> {len(audio_content)} bytes")
        return audio_buffer, content_type
        
    finally:
        # Clean up all chunk files
        for chunk_file in chunk_audio_files:
            try:
                os.remove(chunk_file)
            except:
                pass
        
        # Clean up final output
        try:
            os.remove(output_path)
        except:
            pass

