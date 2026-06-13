# Civora AI

Civora AI is an emergency response assistant for Pakistan that helps users get fast, practical guidance during accidents, floods, medical emergencies, and rescue situations. The app supports text chat, voice input, image-based emergency analysis, location-aware responses, and a document-backed RAG knowledge base for emergency guidance.

## Live Demo

Public app link:

https://civora-ai-seven.vercel.app/frontend/chat.html

Health check:

https://civora-ai-seven.vercel.app/api/health

## What Civora AI Does

- Provides emergency first-aid and rescue guidance in simple Roman Urdu.
- Handles text-based emergency questions through an AI chat interface.
- Supports voice input using Groq Whisper transcription.
- Supports image analysis for emergency scenes using Groq vision models.
- Uses location data to provide context-aware emergency assistance.
- Includes a RAG knowledge base from local emergency documents.
- Falls back to Groq-powered direct chat on Vercel for stable public deployment.

## Tech Stack

- Backend: FastAPI
- Deployment: Vercel Python serverless functions
- LLM: Groq Llama models
- Speech: Groq Whisper
- Vision: Groq vision model
- Embeddings: Hugging Face Inference API
- Vector Store: Chroma
- Frontend: HTML, CSS, JavaScript

## Architecture

```text
User
  -> Frontend chat UI
  -> FastAPI backend
  -> Groq LLM / Whisper / Vision APIs
  -> Optional RAG pipeline with Hugging Face embeddings and Chroma
  -> Emergency response in Roman Urdu
```

On Vercel, the app runs in a stable fallback chat mode by default. This avoids serverless filesystem and cold-start limitations while keeping the public demo usable for everyone. RAG can be enabled later with `ENABLE_RAG_ON_VERCEL=true`.

## Main Features

### Emergency Chat

Users can ask questions such as:

```text
Agar accident ho jaye to pehla step kya hai?
```

The assistant returns practical emergency guidance in Roman Urdu.

### Multi-Modal Support

The backend supports:

- Text messages
- Audio upload for transcription
- Image upload for visual emergency analysis
- Latitude and longitude for location-aware responses

### Public Deployment

The project is deployed on Vercel and can be accessed publicly:

https://civora-ai-seven.vercel.app/frontend/chat.html

## API Endpoints

### Health

```http
GET /api/health
```

Returns backend status, API key availability, RAG status, and fallback status.

### Chat

```http
POST /api/chat
Content-Type: application/json

{
  "message": "Agar bleeding ho rahi ho to kya karna chahiye?"
}
```

### Multi-Modal Chat

```http
POST /api/chat/multi-modal
Content-Type: multipart/form-data
```

Supported fields:

- `message`
- `latitude`
- `longitude`
- `audio_file`
- `image_file`

### Maps Config

```http
GET /api/maps-config
```

Returns map provider configuration for the frontend.

## Environment Variables

Required:

```env
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_api_key
```

Optional:

```env
ENABLE_RAG_ON_VERCEL=false
TOMTOM_API_KEY=your_tomtom_key
GEMINI_API_KEY=your_gemini_key
```

## Local Development

### 1. Clone the Repository

```bash
git clone https://github.com/Anees804804/Civora-Ai.git
cd Civora-Ai
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_api_key
```

### 5. Run the App

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/frontend/chat.html
```

## Project Structure

```text
Civora-Ai/
  main.py                 FastAPI backend
  frontend/               Public frontend pages
  documents/              Emergency knowledge base documents
  requirements.txt        Python dependencies
  vercel.json             Vercel deployment config
  .env.example            Environment variable template
  .gitignore              Git ignore rules
```

## Deployment

The project is deployed to Vercel from GitHub.

Production URL:

https://civora-ai-seven.vercel.app/frontend/chat.html

To redeploy manually:

```bash
vercel --prod --yes
```

## Repository

GitHub:

https://github.com/Anees804804/Civora-Ai

## Status

Production-ready public demo. The Vercel deployment is configured to run reliably in fallback chat mode with Groq, while keeping RAG support available for environments where persistent or writable vector storage is enabled.
