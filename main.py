import os
import base64
import requests
import traceback
import logging
from pathlib import Path
from typing import Optional, List, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

hf_api_key = os.getenv("HF_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    logger.info("GROQ_API_KEY found in environment.")
else:
    logger.warning("⚠️ GROQ_API_KEY is not set. Groq LLM will not initialize correctly without it.")

if hf_api_key:
    logger.info("HF_API_KEY found in environment.")
else:
    logger.info("HF_API_KEY not set. Embeddings will use a fallback provider if available.")

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
fallback_chain = None
retriever = None
groq_llm = None
groq_client = None
rag_initialization_status = {"status": "pending", "error": None}

VECTOR_STORE_DIR = Path("vector_store/chroma")
DOCUMENT_FOLDERS = [Path("documents"), Path("data"), Path("docs")]

def normalize_response(result: Any) -> str:
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def get_embeddings_provider():
    import os
    hf_token = os.getenv("HF_API_KEY")
    if not hf_token:
        raise RuntimeError("HF_API_KEY environment variable is missing")
        
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token, 
        model_name="sentence-transformers/all-small-mpnet-base-v2"
      )


def find_document_roots() -> List[Path]:
    roots = [folder for folder in DOCUMENT_FOLDERS if folder.exists() and folder.is_dir()]
    if not roots:
        default_folder = Path("documents")
        default_folder.mkdir(parents=True, exist_ok=True)
        roots = [default_folder]
        logger.info(f"Created default document folder: {default_folder.resolve()}")
    return roots


def load_documents() -> List[Any]:
    docs: List[Any] = []
    document_roots = find_document_roots()
    supported_extensions = {".pdf", ".txt"}

    for folder in document_roots:
        logger.info(f"Scanning documents from: {folder.resolve()}")
        for source_file in sorted(folder.rglob("*")):
            if source_file.suffix.lower() not in supported_extensions:
                continue
            try:
                if source_file.suffix.lower() == ".pdf":
                    logger.info(f"Loading PDF: {source_file}")
                    loader = PyPDFLoader(str(source_file))
                    docs.extend(loader.load())
                elif source_file.suffix.lower() == ".txt":
                    logger.info(f"Loading TXT: {source_file}")
                    loader = TextLoader(str(source_file), encoding="utf-8")
                    docs.extend(loader.load())
            except Exception as err:
                logger.error(f"Error loading {source_file}: {err}")
                logger.error(traceback.format_exc())

    return docs


def build_rag_chain(retriever_obj: Any, llm_obj: Any) -> Any:
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

    def format_docs(documents: List[Any]) -> str:
        return "\n\n".join(doc.page_content for doc in documents)

    return (
        {"context": retriever_obj | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_obj
        | StrOutputParser()
    )


def build_fallback_chain(llm_obj: Any) -> Any:
    system_prompt = (
        "You are Civora AI, an expert emergency relief and rescue assistant in Pakistan.\n"
        "If no external document context is available, answer the user's emergency assistance query directly and honestly.\n"
        "Provide actionable guidance in Roman Urdu.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return {"question": RunnablePassthrough()} | prompt | llm_obj | StrOutputParser()


def invoke_groq_fallback(user_message: str) -> str:
    system_prompt = (
        "You are Civora AI, an expert emergency relief and rescue assistant in Pakistan.\n"
        "Answer the user's emergency assistance query directly in Roman Urdu using general emergency knowledge.\n"
        "Keep the answer concise, helpful, and easy to understand.\n"
    )

    if groq_llm is not None:
        try:
            ai_response = groq_llm.invoke([
                ("system", system_prompt),
                ("human", user_message),
            ])
            return normalize_response(ai_response)
        except Exception:
            logger.warning("Groq LLM invoke failed, trying direct Groq API client fallback.", exc_info=True)

    if groq_client is not None:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
            )
            if getattr(response, "choices", None):
                first_choice = response.choices[0]
                if hasattr(first_choice, "message"):
                    return first_choice.message.content
                if hasattr(first_choice, "text"):
                    return first_choice.text
            return str(response)
        except Exception:
            logger.error("Direct Groq API fallback failed.", exc_info=True)
            raise

    raise RuntimeError("No Groq fallback available")


def initialize_rag() -> None:
    global rag_chain, fallback_chain, retriever, groq_llm, groq_client, rag_initialization_status

    try:
        logger.info("⏳ RAG initialization started...")
        logger.info(f"GROQ_API_KEY present: {bool(groq_api_key)}")
        logger.info(f"HF_API_KEY present: {bool(hf_api_key)}")
        logger.info(f"OPENAI_API_KEY present: {bool(openai_api_key)}")
        logger.info(f"GOOGLE API credentials present: {bool(google_api_key)}")

        if groq_api_key:
            groq_client = Groq(api_key=groq_api_key)
            groq_llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.3,
                api_key=groq_api_key,
            )
            fallback_chain = build_fallback_chain(groq_llm)
        else:
            groq_client = None
            groq_llm = None
            fallback_chain = None
            rag_initialization_status = {
                "status": "failed",
                "error": "GROQ_API_KEY not set; Groq LLM cannot be initialized.",
            }
            logger.error("GROQ_API_KEY missing; cannot initialize Groq LLM. Please set GROQ_API_KEY.")
            return

        embeddings = get_embeddings_provider()

        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        vector_store_exists = any(VECTOR_STORE_DIR.iterdir())
        logger.info(f"Vector store directory: {VECTOR_STORE_DIR.resolve()}, exists: {vector_store_exists}")

        # Decide whether to (re)index on startup. Helpful for ephemeral hosts like Vercel
        force_reindex = os.getenv("REINDEX_ON_STARTUP", "false").lower() == "true"
        running_on_vercel = bool(os.getenv("VERCEL"))
        should_reindex = force_reindex or running_on_vercel or not vector_store_exists

        if should_reindex:
            logger.info(f"Reindexing vector store on startup (force_reindex={force_reindex}, vercel={running_on_vercel}).")
            docs = load_documents()
            if not docs:
                rag_initialization_status = {
                    "status": "warning",
                    "error": "No source documents found. RAG chain will not initialize.",
                }
                logger.warning("No source documents found. RAG chain will not initialize.")
                return

            logger.info(f"Total documents loaded: {len(docs)}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            logger.info(f"Total chunks created: {len(splits)}")

            logger.info("Creating vector store from documents with Chroma...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(VECTOR_STORE_DIR),
            )
            logger.info("Vector store created successfully (reindexed).")
        else:
            # Attempt to load existing store; if it fails, fall back to rebuilding from documents
            try:
                logger.info("Loading existing Chroma vector store.")
                vectorstore = Chroma(persist_directory=str(VECTOR_STORE_DIR), embedding_function=embeddings)
            except Exception:
                logger.warning("Failed to load existing Chroma store; attempting to rebuild from documents.", exc_info=True)
                docs = load_documents()
                if not docs:
                    rag_initialization_status = {
                        "status": "warning",
                        "error": "No source documents found. RAG chain will not initialize.",
                    }
                    logger.warning("No source documents found. RAG chain will not initialize.")
                    return

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)
                logger.info(f"Total chunks created (rebuild): {len(splits)}")
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    persist_directory=str(VECTOR_STORE_DIR),
                )
                logger.info("Vector store rebuilt successfully after load failure.")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_chain = build_rag_chain(retriever, groq_llm)

        rag_initialization_status = {"status": "success", "error": None}
        logger.info("✅ Civora Enhanced Multi-Modal Backend Ready!")

    except Exception as exc:
        logger.error("RAG initialization failed.", exc_info=True)
        rag_chain = None
        retriever = None
        rag_initialization_status = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

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
            "groq": "set" if groq_api_key else "not_set",
            "huggingface": "set" if hf_api_key else "not_set",
            "openai": "set" if openai_api_key else "not_set",
            "google": "set" if google_api_key else "not_set",
        },
        "fallback_available": fallback_chain is not None,
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
    global rag_chain, fallback_chain

    user_message = req.message
    if rag_chain is not None:
        try:
            ai_response = rag_chain.invoke(user_message)
            return {
                "response": normalize_response(ai_response),
                "used_fallback": False,
            }
        except Exception:
            logger.error("RAG chain failed during chat handling; routing to Groq direct fallback.", exc_info=True)

    if fallback_chain is not None:
        try:
            ai_response = fallback_chain.invoke(user_message)
            return {
                "response": normalize_response(ai_response),
                "used_fallback": True,
            }
        except Exception:
            logger.error("Fallback chain failed during chat handling; routing to Groq direct fallback.", exc_info=True)

    try:
        groq_response = invoke_groq_fallback(user_message)
        return {
            "response": groq_response,
            "used_fallback": True,
        }
    except Exception:
        logger.error("Groq direct fallback also failed in chat endpoint.", exc_info=True)
        return {
            "response": "Maaf kijiye, abhi system temporarily unavailable hai. Please try again shortly.",
            "used_fallback": True,
        }

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
            chain = rag_chain if rag_chain is not None else fallback_chain
            if chain is None:
                error_detail = rag_initialization_status.get("error", "Unknown error")
                logger.error(f"No chain available in multi-modal endpoint: {rag_initialization_status}")
                return {
                    "response": (
                        f"System unavailable: {error_detail}. Check /api/health endpoint for details or add documents to documents/ folder."
                    ),
                    "location": detected_address,
                }

            try:
                ai_response = chain.invoke(final_prompt_with_meta)
                return {
                    "response": normalize_response(ai_response),
                    "location": detected_address,
                    "used_fallback": rag_chain is None,
                }
            except Exception as exc:
                logger.error("Error invoking chat chain.", exc_info=True)
                if rag_chain is not None and fallback_chain is not None:
                    try:
                        fallback_response = fallback_chain.invoke(final_prompt_with_meta)
                        return {
                            "response": normalize_response(fallback_response),
                            "location": detected_address,
                            "used_fallback": True,
                        }
                    except Exception:
                        logger.error("Fallback chain failed in multi-modal endpoint.", exc_info=True)
                return {"response": "Server error processing your request.", "location": detected_address}

        return {"response": "Aapka input khaali mila.", "location": detected_address}

    except Exception as e:
        logger.error("Multi-modal endpoint error.", exc_info=True)
        return {"response": f"Server error: {e}", "location": detected_address}
