import os
import base64
import requests

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("Please set the GROQ_API_KEY environment variable before starting the app.")
os.environ["GROQ_API_KEY"] = groq_api_key

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# RAG Tools, Loaders aur Vector Store
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

app = FastAPI(title="Civora AI - Enhanced Multi-Modal Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/frontend/chat.html")

# Global instances
rag_chain = None
retriever = None
groq_client = Groq()

def initialize_rag():
    global rag_chain, retriever
    try:
        print("⏳ Documents load aur embed ho rahe hain...")
        docs = []
        folder_path = "documents"
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if file.endswith(".pdf"):
                pdf_loader = PyPDFLoader(full_path)
                docs.extend(pdf_loader.load())
            elif file.endswith(".txt"):
                txt_loader = TextLoader(full_path, encoding="utf-8")
                docs.extend(txt_loader.load())

        if not docs:
            print("⚠️ Warning: 'documents' folder khali hai!")
            return
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        
        system_prompt = (
            "You are Civora AI, an expert emergency relief and rescue assistant in Pakistan.\n"
            "Use the provided document context and user location metadata to guide them.\n"
            "Provide step-by-step first-aid guidelines, nearest hospital guidance, and relevant rescue numbers (Rescue 1122, Fire Brigade 16, etc.).\n"
            "CRITICAL: Always respond in natural, easy-to-read Roman Urdu (Urdu language written in English alphabets).\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)
            
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        print("✅ Civora Enhanced Multi-Modal Backend Ready!")
    except Exception as e:
        print(f"❌ Setup error: {e}")

@app.on_event("startup")
async def startup_event():
    initialize_rag()

# 📍 1. MAPS CONFIG ENDPOINT
@app.get("/api/maps-config")
async def get_maps_config():
    return {"status": "active", "provider": "open-street-maps"}

# 📝 2. STANDARD TEXT CHAT ENDPOINT
class ChatReq(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(req: ChatReq):
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG system initialized nahi hua.")
    try:
        ai_response = rag_chain.invoke(req.message)
        return {"response": ai_response}
    except Exception as e:
        return {"response": f"Server error: {str(e)}"}

# 🎙️ 📸 📍 3. MULTI-MODAL CORE ENDPOINT (Enhanced Location Handling)
@app.post("/api/chat/multi-modal")
async def multi_modal_endpoint(
    message: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    global rag_chain, retriever
    user_query = message or ""
    location_string = ""
    detected_address = "Unknown Location"

    # ENHANCEMENT: Reverse Geocoding with OpenStreetMap (Free, No Key Needed)
    if latitude and longitude:
        try:
            geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
            headers = {'User-Agent': 'CivoraAI-Hackathon-App'}
            geo_res = requests.get(geo_url, headers=headers).json()
            detected_address = geo_res.get("display_name", f"Lat {latitude}, Lng {longitude}")
        except Exception:
            detected_address = f"Lat {latitude}, Lng {longitude}"
            
        location_string = f"\n\n[USER METADATA - Detected Location: {detected_address}. Coordinates: Lat {latitude}, Lng {longitude}. User ko batayein ke unki location track ho chuki hai aur qareebi medical shelter/hospital ka rasta Civora Maps widget par load ho chuka hai. Guide them gently in Roman Urdu.]"

    try:
        # A. VOICE INPUT PROCESSING (Whisper)
        if audio_file:
            print("🎙️ Audio file processing...")
            audio_bytes = await audio_file.read()
            audio_path = f"temp_{audio_file.filename}"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
                
            with open(audio_path, "rb") as file_to_transcribe:
                translation = groq_client.audio.transcriptions.create(
                    file=file_to_transcribe,
                    model="whisper-large-v3",
                    prompt="Urdu, Roman Urdu emergency incident reporting first aid"
                )
            user_query = translation.text
            os.remove(audio_path)
            print(f"✅ Voice Transcribed: {user_query}")

        final_prompt_with_meta = f"{user_query} {location_string}"

        # B. OCR / IMAGE VISION PROCESSING
        if image_file:
            print("📸 Processing incoming image for analysis...")
            image_bytes = await image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            context_text = ""
            if retriever and user_query:
                docs = retriever.invoke(user_query)
                context_text = "\n".join([d.page_content for d in docs])

            response = groq_client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": f"Analyze this emergency image/scene. User Query context: {final_prompt_with_meta}. Medical/Document context: {context_text}. Provide immediate response in clean Roman Urdu script only."
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0.2
            )
            return {"response": response.choices[0].message.content, "location": detected_address}

        # C. STANDARD OR VOICE-BASED RAG INFERENCE
        if user_query or location_string:
            if rag_chain:
                ai_response = rag_chain.invoke(final_prompt_with_meta)
                return {"response": ai_response, "location": detected_address}
            else:
                return {"response": "System RAG chain initialize nahi ho saka.", "location": detected_address}

        return {"response": "Aapka input khaali mila.", "location": detected_address}

    except Exception as e:
        return {"response": f"Server error: {str(e)}", "location": detected_address}