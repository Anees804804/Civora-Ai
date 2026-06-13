# Civora AI - Emergency & Flood Relief Assistant

Advanced multi-modal emergency response system with AI-powered first aid guidance, flood tracking, and medical camp location assistance for Pakistan.

## Features

- 🎤 **Voice Input**: Direct voice commands via Whisper transcription
- 📸 **Image Analysis**: Vision-based emergency scene analysis
- 📍 **Location Tracking**: Real-time GPS integration with reverse geocoding
- 🤖 **RAG System**: Context-aware responses from knowledge base
- 🌊 **Flood Monitoring**: Real-time water level tracking
- 🏥 **Medical Assistance**: Hospital and rescue service finder
- 🔴 **Emergency Contacts**: Direct dialer integration (1122 Rescue, 115 Edhi)

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **LLM**: Groq Llama 3.1 + Vision
- **Speech**: Whisper (Groq)
- **Embeddings**: Hugging Face Sentence Transformers
- **Vector Store**: Chroma
- **Frontend**: HTML5 + Tailwind CSS + Material Design
- **Deployment**: Vercel (Python support)

## Local Setup

### Prerequisites
- Python 3.10+
- Groq API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/Anees804804/Civora-Ai.git
cd Civora-Ai

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variable
$env:GROQ_API_KEY = "your_groq_api_key"

# Run the server
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Access
- Frontend: `http://127.0.0.1:8000/`
- API Docs: `http://127.0.0.1:8000/docs`

## Vercel Deployment

### Setup Steps

1. **Connect Repository**
   - Go to [Vercel Dashboard](https://vercel.com)
   - Import your GitHub repository

2. **Set Environment Variables**
   - In Vercel project settings, add:
     - `GROQ_API_KEY`: Your Groq API key

3. **Deploy**
   - Vercel automatically detects `vercel.json` and deploys
   - Your app will be live at `https://your-project-name.vercel.app`

## API Endpoints

### Chat Endpoint
```
POST /api/chat
Content-Type: application/json

{
  "message": "Your emergency query"
}
```

### Multi-Modal Endpoint
```
POST /api/chat/multi-modal
Content-Type: multipart/form-data

- message: (optional) text query
- latitude: (optional) user location latitude
- longitude: (optional) user location longitude
- audio_file: (optional) voice message
- image_file: (optional) emergency scene image
```

## Directory Structure

```
Civora-Ai/
├── main.py                 # FastAPI backend
├── frontend/               # HTML frontend
│   ├── chat.html          # AI assistant interface
│   ├── healthcare.html    # Emergency contacts
│   ├── emergency.html     # Flood alerts
│   └── index.html         # Home page
├── documents/             # RAG knowledge base
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel configuration
├── .env.example          # Environment variables template
└── .gitignore            # Git ignore rules
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## License

This project is open source and available under the MIT License.

## Support

For issues or contributions, please visit: https://github.com/Anees804804/Civora-Ai

---

**Status**: ✅ Ready for production deployment
