import os
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Configuración inicial
load_dotenv()
app = FastAPI()

# Configuración de permisos (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Conexión a Servicios
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not supabase_url or not google_api_key:
    raise ValueError("❌ Error: Faltan variables de entorno.")

try:
    supabase: Client = create_client(supabase_url, supabase_key)
    genai.configure(api_key=google_api_key)
except Exception as e:
    print(f"❌ Error conexión servicios: {e}")

class UserQuery(BaseModel):
    pregunta: str
    session_id: str = "anonimo"

# 3. Función de búsqueda (Embeddings)
def buscar_contexto(pregunta_usuario: str):
    try:
        # IMPORTANTE: Mantenemos el modelo de embeddings 004 que es universal
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=pregunta_usuario,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']

        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.4,
            "match_count": 3
        }).execute()
        
        contexto = "\n\n".join([item['content'] for item in response.data])
        return contexto
    except Exception as e:
        print(f"⚠️ Error buscando contexto: {e}")
        return ""

# 4. Chat Endpoint
@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    print(f"📩 Pregunta recibida: {query.pregunta}")

    saludos = ["hola", "buen dia", "buenas", "que tal"]
    if any(s in query.pregunta.lower() for s in saludos) and len(query.pregunta) < 20:
        return {"respuesta": "¡Hola! 👋 Soy UniBot. ¿En qué puedo ayudarte con Alumnado?"}

    contexto = buscar_contexto(query.pregunta)
    
    prompt = f"""
    Eres UniBot de UNCAUS. Responde usando SOLO este contexto:
    "{contexto}"
    Si no está en el contexto, di que no sabes.
    """ 

    try:
        # --- CAMBIO CRITICO: USAMOS EL MODELO QUE SÍ TIENES ---
        # Tu cuenta tiene acceso a gemini-2.0-flash
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        response = model.generate_content(prompt)
        respuesta_final = response.text
    except Exception as e:
        print(f"⚠️ Falló Gemini 2.0: {e}")
        try:
            # Si falla, intentamos con 'gemini-2.0-flash-lite' que también te sale en la lista
            model_backup = genai.GenerativeModel('gemini-2.0-flash-lite')
            response = model_backup.generate_content(prompt)
            respuesta_final = response.text
        except Exception as e2:
            respuesta_final = "Lo siento, hubo un error técnico. (Modelo no compatible)"
            print(f"❌ Error Gemini Crítico: {e2}")

    # Guardar log
    try:
        supabase.table("chat_logs").insert({
            "session_id": query.session_id,
            "user_input": query.pregunta,
            "bot_response": respuesta_final
        }).execute()
    except:
        pass

    return {"respuesta": respuesta_final}

@app.get("/")
def home():
    return {"status": "UniBot v4.0 - MODELO 2.0 ACTIVADO 🚀"}