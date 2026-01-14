import asyncio
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from . import models
from . import security
from . import tts_logic
from . import stt_logic
import io

app = FastAPI(
    title="Mac Dia Server - OpenAI TTS Compatible API",
    description="Provides a TTS endpoint using mlx-audio backend.",
    version="0.1.0",
)

# Enable CORS for local HTML file access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (file://, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/v1/audio/speech",
    response_description="Audio stream in the requested format",
    tags=["TTS"],
)
async def create_speech(request: models.TTSRequest):
    """Handles the text-to-speech request, compatible with OpenAI's API."""
    try:
        # Run the potentially blocking TTS generation in a separate thread with 60s timeout
        audio_buffer, content_type = await asyncio.wait_for(
            asyncio.to_thread(
                tts_logic.generate_speech_from_text_sync, request
            ),
            timeout=60.0
        )

        # Reset buffer position to the beginning before streaming
        audio_buffer.seek(0)

        return StreamingResponse(audio_buffer, media_type=content_type)

    except asyncio.TimeoutError:
        print("TTS generation timed out after 60 seconds")
        raise HTTPException(
            status_code=408,
            detail="Request timed out after 60 seconds"
        )
    except Exception as e:
        # Basic error handling, might need refinement
        print(f"Error during TTS generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate audio: {str(e)}"
        )


@app.post(
    "/v1/audio/speech/clone",
    response_description="Audio stream with cloned voice",
    tags=["TTS"],
)
async def create_cloned_speech(
    input: str = Form(..., description="The text to synthesize"),
    ref_audio: UploadFile = File(..., description="Reference audio file for voice cloning (~10 seconds recommended)"),
    ref_text: Optional[str] = Form(None, description="Transcript of the reference audio (optional, will auto-transcribe if not provided)"),
    response_format: Optional[str] = Form("mp3", description="Output audio format: mp3, wav, opus, aac, flac"),
    speed: Optional[float] = Form(1.0, description="Speech speed (0.25 to 4.0)"),
):
    """
    Generates speech using voice cloning from a reference audio file.
    
    Upload a reference audio file (~10 seconds of clear speech recommended) 
    and the text you want synthesized in that voice.
    """
    import tempfile
    import os
    
    # Validate speed
    if speed < 0.25 or speed > 4.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.25 and 4.0")
    
    # Validate response format
    valid_formats = ["mp3", "opus", "aac", "flac", "wav"]
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Supported formats: {', '.join(valid_formats)}"
        )
    
    # Save uploaded reference audio to temp file
    ref_audio_content = await ref_audio.read()
    
    # Get file extension from uploaded file
    ref_ext = ref_audio.filename.split(".")[-1].lower() if ref_audio.filename and "." in ref_audio.filename else "mp3"
    
    # Create temp file for reference audio
    temp_ref_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ref_ext}")
    temp_ref_path = temp_ref_file.name
    
    try:
        # Write reference audio to temp file
        temp_ref_file.write(ref_audio_content)
        temp_ref_file.close()
        
        print(f"Saved reference audio to: {temp_ref_path} ({len(ref_audio_content)} bytes)")
        
        # Run voice cloning TTS with 120s timeout (voice cloning takes longer)
        audio_buffer, content_type = await asyncio.wait_for(
            asyncio.to_thread(
                tts_logic.generate_cloned_speech_sync,
                text=input,
                ref_audio_path=temp_ref_path,
                ref_text=ref_text,
                output_format=response_format,
                speed=speed
            ),
            timeout=120.0
        )
        
        audio_buffer.seek(0)
        return StreamingResponse(audio_buffer, media_type=content_type)
    
    except asyncio.TimeoutError:
        print("Voice cloning timed out after 120 seconds")
        raise HTTPException(
            status_code=408,
            detail="Voice cloning request timed out after 120 seconds"
        )
    except Exception as e:
        print(f"Error during voice cloning: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate cloned audio: {str(e)}"
        )
    finally:
        # Clean up temp reference audio file
        try:
            os.remove(temp_ref_path)
        except Exception as e:
            print(f"Warning: Could not delete temp reference file {temp_ref_path}: {e}")

@app.post(
    "/v1/audio/transcriptions",
    dependencies=[Depends(security.get_api_key)],  # Apply API Key authentication
    tags=["STT"],
)
async def create_transcription(
    file: UploadFile = File(...),
    model: Optional[str] = Form("mlx-community/whisper-large-v3-turbo"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """Handles the speech-to-text request, compatible with OpenAI's API."""
    try:
        # Read file content
        file_content = await file.read()

        # Create memory buffer from file content
        audio_buffer = io.BytesIO(file_content)

        # Validate supported formats
        valid_formats = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
        file_ext = file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else ""

        if file_ext not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats are: {', '.join(valid_formats)}"
            )

        # Run the potentially blocking STT processing in a separate thread with 60s timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                stt_logic.transcribe_audio_sync,
                audio_buffer,
                model_name="mlx-community/whisper-large-v3-turbo",
                language=language,
                prompt=prompt,
                temperature=temperature,
            ),
            timeout=60.0
        )

        # Format the response according to the requested format
        # For now, we only support JSON response format
        if response_format == "json" or response_format == "verbose_json":
            return result
        else:
            # For text, srt, vtt formats, we would need to implement those conversions
            # For now, just return the text
            return {"text": result["text"]}

    except asyncio.TimeoutError:
        print("STT processing timed out after 60 seconds")
        raise HTTPException(
            status_code=408,
            detail="Request timed out after 60 seconds"
        )
    except Exception as e:
        print(f"Error during STT processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe audio: {str(e)}"
        )



# Optional: Add a root endpoint for basic health check or info
@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Mac Dia Server is running. Use POST /v1/audio/speech for TTS."}

# 添加这个函数作为入口点
def start_server():
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
