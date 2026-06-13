import os
import base64
import requests
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from groq import Groq

# Get HuggingFace API key for embeddings (free tier available)
hf_api_key = os.getenv("HF_API_KEY")
if not hf_api_key:
    logger.warning("⚠️ HF_API_KEY not set. Get free API key from https://huggingface.co/settings/tokens")
    logger.info("Using lightweight embeddings fallback.")

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
rag_initialization_status = {"status": "pending", "error": None}

def initialize_rag():
    global rag_chain, retriever, rag_initialization_status
    try:
        logger.info("⏳ RAG initialization started...")
        logger.info(f"GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}")
        logger.info(f"HF_API_KEY present: {bool(hf_api_key)}")
        
        docs = []
        folder_path = "documents"
        
        logger.info(f"Checking for documents folder at: {os.path.abspath(folder_path)}")
        if not os.path.exists(folder_path):
            logger.warning(f"Documents folder does not exist. Creating it at {os.path.abspath(folder_path)}")
            os.makedirs(folder_path)
        
        files_found = os.listdir(folder_path)
        logger.info(f"Files in documents folder: {files_found}")
        
        for file in files_found:
            full_path = os.path.join(folder_path, file)
            logger.info(f"Processing file: {file}")
            try:
                if file.endswith(".pdf"):
                    logger.info(f"Loading PDF: {full_path}")
                    pdf_loader = PyPDFLoader(full_path)
                    docs.extend(pdf_loader.load())
                    logger.info(f"Successfully loaded {len(docs)} documents from {file}")
                elif file.endswith(".txt"):
                    logger.info(f"Loading TXT: {full_path}")
                    txt_loader = TextLoader(full_path, encoding="utf-8")
                    docs.extend(txt_loader.load())
                    logger.info(f"Successfully loaded {len(docs)} documents from {file}")
            except Exception as file_err:
                logger.error(f"Error loading {file}: {str(file_err)}")
                logger.error(traceback.format_exc())

        if not docs:
            rag_initialization_status = {"status": "warning", "error": "No documents found in documents folder. RAG will not function."}
            logger.warning("⚠️ No documents found in documents folder!")
            logger.info("To enable RAG, add PDF or TXT files to the 'documents' folder.")
            return
        
        logger.info(f"Total documents loaded: {len(docs)}")
        
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        logger.info(f"Total chunks created: {len(splits)}")
        
        # Use HuggingFace Inference API for embeddings (free, no local models)
        logger.info("Initializing embeddings...")
        if hf_api_key:
            logger.info("Using HuggingFace Endpoint Embeddings")
            embeddings = HuggingFaceEndpointEmbeddings(
                model="https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=hf_api_key,
                task="feature-extraction"
            )
        else:
            # Fallback: Simple hash-based embeddings (lightweight)
            logger.info("Using Fake Embeddings fallback")
            from langchain_community.embeddings.fake import FakeEmbeddings
            embeddings = FakeEmbeddings(model_name="fake-model")
        
        logger.info("Creating vector store with Chroma...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("Vector store created successfully")
        
        logger.info("Initializing Groq LLM...")
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
            
        logger.info("Building RAG chain...")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        
        rag_initialization_status = {"status": "success", "error": None}
        logger.info("✅ Civora Enhanced Multi-Modal Backend Ready!")
    except Exception as e:
        error_msg = f"❌ Setup error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        rag_initialization_status = {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}
        rag_chain = None
        retriever = None

@app.on_event("startup")
async def startup_event():
    initialize_rag()

# � HEALTH CHECK ENDPOINT
@app.get("/api/health")
async def health_check():
    return {
        "status": "running",
        "rag_initialization": rag_initialization_status,
        "api_keys": {
            "groq": "set" if os.getenv("GROQ_API_KEY") else "not_set",
            "huggingface": "set" if hf_api_key else "not_set"
        }
    }

# �📍 1. MAPS CONFIG ENDPOINT
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
        error_detail = rag_initialization_status.get("error", "Unknown error")
        logger.error(f"RAG chain is None. Initialization status: {rag_initialization_status}")
        raise HTTPException(
            status_code=503, 
            detail=f"RAG system initialize nahi ho saka: {error_detail}. Check logs or add documents to documents/ folder."
        )
    try:
        ai_response = rag_chain.invoke(req.message)
        return {"response": ai_response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
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
            logger.info("🎙️ Audio file processing...")
            audio_bytes = await audio_file.read()
            audio_path = f"temp_{audio_file.filename}"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
                
            try:
                with open(audio_path, "rb") as file_to_transcribe:
                    translation = groq_client.audio.transcriptions.create(
                        file=file_to_transcribe,
                        model="whisper-large-v3",
                        prompt="Urdu, Roman Urdu emergency incident reporting first aid"
                    )
                user_query = translation.text
                logger.info(f"✅ Voice Transcribed: {user_query}")
            except Exception as audio_err:
                logger.error(f"Audio transcription error: {str(audio_err)}")
                user_query = message or ""
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)

        final_prompt_with_meta = f"{user_query} {location_string}"

        # B. OCR / IMAGE VISION PROCESSING
        if image_file:
            logger.info("📸 Processing incoming image for analysis...")
            try:
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
                logger.info("✅ Image analysis complete")
                return {"response": response.choices[0].message.content, "location": detected_address}
            except Exception as img_err:
                logger.error(f"Image processing error: {str(img_err)}\n{traceback.format_exc()}")
                return {"response": f"Image analysis error: {str(img_err)}", "location": detected_address}

        # C. STANDARD OR VOICE-BASED RAG INFERENCE
        if user_query or location_string:
            if rag_chain:
                logger.info(f"Invoking RAG chain with query: {user_query[:50]}...")
                ai_response = rag_chain.invoke(final_prompt_with_meta)
                logger.info("✅ RAG response generated")
                return {"response": ai_response, "location": detected_address}
            else:
                error_detail = rag_initialization_status.get("error", "Unknown error")
                logger.error(f"RAG chain not initialized: {rag_initialization_status}")
                return {
                    "response": f"System RAG chain initialize nahi ho saka: {error_detail}. Check /api/health endpoint for details or add documents to documents/ folder.",
                    "location": detected_address
                }

        return {"response": "Aapka input khaali mila.", "location": detected_address}

    except Exception as e:
        logger.error(f"Multi-modal endpoint error: {str(e)}\n{traceback.format_exc()}")
        return {"response": f"Server error: {str(e)}", "location": detected_address}