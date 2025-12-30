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
    raise ValueError("❌ Error: Faltan las variables de entorno.")

# Inicializamos clientes
try:
    supabase: Client = create_client(supabase_url, supabase_key)
    genai.configure(api_key=google_api_key)
except Exception as e:
    print(f"❌ Error al conectar servicios: {e}")

# Modelo de datos
class UserQuery(BaseModel):
    pregunta: str
    session_id: str = "anonimo"

# 3. Función para buscar información
def buscar_contexto(pregunta_usuario: str):
    try:
        # A. Embedding
        # NOTA: Para embeddings, 'models/' suele ser necesario, pero para chat no.
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=pregunta_usuario,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']

        # B. Búsqueda en Supabase
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.4, # Lo bajé un poquito para que encuentre más cosas
            "match_count": 3
        }).execute()
        
        # C. Unir texto
        contexto = "\n\n".join([item['content'] for item in response.data])
        return contexto
        
    except Exception as e:
        print(f"⚠️ Advertencia buscando contexto: {e}")
        return ""

# 4. El Cerebro del Chat
@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    print(f"📩 Pregunta recibida: {query.pregunta}")

    # --- DETECTOR DE SALUDOS ---
    saludos = ["hola", "buen dia", "buen día", "buenas", "que tal", "hello", "hi"]
    mensaje_usuario = query.pregunta.lower().strip()
    
    if any(s in mensaje_usuario for s in saludos) and len(mensaje_usuario) < 20:
        return {"respuesta": "¡Hola! 👋 Soy UniBot, el asistente virtual de Alumnado UNCAUS. ¿En qué trámite, fecha o requisito puedo ayudarte hoy?"}
    # ---------------------------

    # 1. Buscamos información
    contexto = buscar_contexto(query.pregunta)
    
    # 2. Instrucciones
    prompt = f"""
    Eres UniBot, el asistente virtual de la UNCAUS.
    Responde la pregunta del usuario basándote EXCLUSIVAMENTE en el siguiente contexto.

    CONTEXTO RECUPERADO:
    "{contexto}"

    ---
    INSTRUCCIONES:
    1. Usa la información del CONTEXTO para responder.
    2. Si la respuesta NO está en el contexto, di textualmente: "Lo siento, no tengo información sobre ese tema específico en mi base de conocimientos de Alumnado."
    3. Sé amable, breve y usa emojis.
    """ 

    # 3. Generamos la respuesta con Gemini
    try:
        # --- AQUÍ ESTABA EL ERROR ---
        # Cambiamos 'models/gemini-1.5-flash' por 'gemini-1.5-flash'
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(prompt)
        respuesta_final = response.text
    except Exception as e:
        # Si falla Flash, intentamos con Pro como respaldo
        try:
            print(f"⚠️ Falló Flash, intentando con Pro... Error: {e}")
            model_backup = genai.GenerativeModel('gemini-pro')
            response = model_backup.generate_content(prompt)
            respuesta_final = response.text
        except Exception as e2:
            respuesta_final = "Lo siento, hubo un error técnico al conectar con la IA."
            print(f"❌ Error Gemini Crítico: {e2}")

    # 4. Guardamos log
    try:
        supabase.table("chat_logs").insert({
            "session_id": query.session_id,
            "user_input": query.pregunta,
            "bot_response": respuesta_final
        }).execute()
    except Exception as e:
        print(f"⚠️ No se pudo guardar el log: {e}")

    return {"respuesta": respuesta_final}

@app.get("/")
def home():
    return {"UniBot ACTUALIZADO v2 🚀"}
