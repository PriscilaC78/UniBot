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
        # IMPORTANTE: Para embeddings SÍ se usa 'models/'
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
        # Intentamos con Flash (versión estable)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        respuesta_final = response.text
    except Exception as e:
        print(f"⚠️ Falló Flash: {e}")
        try:
            # RESPALDO: Usamos gemini-pro (el clásico) si falla Flash
            model_backup = genai.GenerativeModel('gemini-pro')
            response = model_backup.generate_content(prompt)
            respuesta_final = response.text
        except Exception as e2:
            respuesta_final = "Lo siento, hubo un error técnico al conectar con la IA. (Clave o Modelo inválido)"
            print(f"❌ Error Gemini Crítico: {e2}")

    # Guardar log (si falla no detiene el bot)
    try:
        supabase.table("chat_logs").insert({
            "session_id": query.session_id,
            "user_input": query.pregunta,
            "bot_response": respuesta_final
        }).execute()
    except:
        pass

    return {"respuesta": respuesta_final}

# --- HERRAMIENTA DE DIAGNÓSTICO (NUEVO) ---
@app.get("/test-google")
def test_google_connection():
    """Prueba qué modelos ve realmente el servidor"""
    status = {"api_key_detectada": bool(google_api_key), "modelos_disponibles": [], "error": None}
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                status["modelos_disponibles"].append(m.name)
    except Exception as e:
        status["error"] = str(e)
    return status

@app.get("/")
def home():
    return {"status": "UniBot v3.5 - MODO DIAGNOSTICO 🔧"}