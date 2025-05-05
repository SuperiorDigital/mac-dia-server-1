import asyncio
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse

from . import models
from . import security
from . import tts_logic

app = FastAPI(
    title="Mac Dia Server - OpenAI TTS Compatible API",
    description="Provides a TTS endpoint using mlx-audio backend.",
    version="0.1.0",
)

@app.post(
    "/v1/audio/speech",
    dependencies=[Depends(security.get_api_key)], # Apply API Key authentication
    response_description="Audio stream in the requested format",
    tags=["TTS"],
)
async def create_speech(request: models.TTSRequest):
    """Handles the text-to-speech request, compatible with OpenAI's API."""
    try:
        # Run the potentially blocking TTS generation in a separate thread
        audio_buffer, content_type = await asyncio.to_thread(
            tts_logic.generate_speech_from_text_sync, request
        )

        # Reset buffer position to the beginning before streaming
        audio_buffer.seek(0)

        return StreamingResponse(audio_buffer, media_type=content_type)

    except Exception as e:
        # Basic error handling, might need refinement
        print(f"Error during TTS generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate audio: {str(e)}"
        )

# Optional: Add a root endpoint for basic health check or info
@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Mac Dia Server is running. Use POST /v1/audio/speech for TTS."}

# 添加这个函数作为入口点
def start_server():
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
