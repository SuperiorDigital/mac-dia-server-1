# Mac Dia Server

FastAPI server providing an OpenAI-compatible Text-to-Speech (TTS) API endpoint, utilizing `mlx-audio` for generation on Apple Silicon.

## About Dia-1.6B

Dia-1.6B is a state-of-the-art open-source text-to-speech (TTS) model developed by Nari Labs, featuring 1.6 billion parameters. This service is dedicated to providing TTS capabilities using the fixed model `mlx-community/Dia-1.6B-4bit`. Key features include:

- Multi-speaker dialogue generation using [S1], [S2] tags in the input text
- Fine-grained control over voice, emotion, and speaking style
- Support for non-verbal expressions like laughter, coughing, and more
- Voice cloning capabilities for personalized speech synthesis
- Optimized for English language generation

Dia-1.6B is comparable in performance to leading commercial TTS solutions, while remaining fully open and customizable for research and production use.

## Setup

1.  **Install Dependencies:**
    Requires Python 3.12+ and `uv`.
    ```bash
    uv venv  # Create virtual environment
    source .venv/bin/activate
    uv pip install .
    # Special attention might be needed for installing mlx and mlx-audio.
    # Follow official MLX documentation.
    uv run start.py
    ```

2.  **Configure API Key:**
    Create a `.env` file in the project root:
    ```
    API_KEY=your_actual_api_key
    ```

3.  **Run the Server:**
    ```bash
    uv run start.py
    ```

## API Endpoint

-   **URL:** `/v1/audio/speech`
-   **Method:** `POST`
-   **Authentication:** `Authorization: Bearer <YOUR_API_KEY>`
-   **Request Body:** (See OpenAI TTS API documentation)
    -   `model` (string): e.g., "tts-1"
    -   `input` (string): Text to synthesize.
    -   `voice` (string): e.g., "alloy"
    -   `response_format` (string, optional): e.g., "mp3", defaults to "mp3".
    -   `speed` (float, optional): Speed, defaults to 1.
-   **Response:** Audio stream in the specified format.


## CURl
```bash

  curl -X 'POST' \
  'http://localhost:8000/v1/audio/speech' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer " \
  -d '{
  "model": "string",
  "input": "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face.",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1
 }' \
    --output speech.mp3
```


```
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F file=@/yourfile \
  -F model=mlx-community/whisper-large-v3-turbo \
  -F language=en
```