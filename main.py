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

# Configuración de permisos (CORS) para que funcione en cualquier web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Conexión a Servicios (Supabase y Google)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not supabase_url or not google_api_key:
    raise ValueError("❌ Error: Faltan las variables de entorno en el archivo .env")

# Inicializamos clientes
try:
    supabase: Client = create_client(supabase_url, supabase_key)
    genai.configure(api_key=google_api_key)
except Exception as e:
    print(f"❌ Error al conectar servicios: {e}")

# Modelo de datos que recibimos del usuario
class UserQuery(BaseModel):
    pregunta: str
    session_id: str = "anonimo"

# 3. Función para buscar información en tu PDF (Base de datos)
def buscar_contexto(pregunta_usuario: str):
    try:
        # A. Convertimos la pregunta en números (Embedding)
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=pregunta_usuario,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']

        # B. Buscamos en Supabase los 3 fragmentos más parecidos
        # Optimización: Bajamos match_count a 3 para más velocidad
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.5, # Sensibilidad de búsqueda
            "match_count": 3        # Traer menos texto para ser más rápido
        }).execute()
        
        # C. Unimos los fragmentos en un solo texto
        contexto = "\n\n".join([item['content'] for item in response.data])
        return contexto
        
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudo obtener contexto: {e}")
        return ""

# 4. El Cerebro del Chat
@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    print(f"📩 Pregunta recibida: {query.pregunta}")

    # --- PASO RÁPIDO: DETECTOR DE SALUDOS ---
    # Si saludan, respondemos directo sin buscar en la base de datos (Ahorra tiempo)
    saludos = ["hola", "buen dia", "buen día", "buenas", "que tal", "hello", "hi"]
    mensaje_usuario = query.pregunta.lower().strip()
    
    # Si el mensaje contiene un saludo y es corto (menos de 20 letras)
    if any(s in mensaje_usuario for s in saludos) and len(mensaje_usuario) < 20:
        return {"respuesta": "¡Hola! 👋 Soy UniBot, el asistente virtual de Alumnado UNCAUS. ¿En qué trámite, fecha o requisito puedo ayudarte hoy?"}
    # ----------------------------------------

    # 1. Buscamos información en el PDF
    contexto = buscar_contexto(query.pregunta)
    
    # 2. Instrucciones para la Inteligencia Artificial
    prompt = f"""
    Eres UniBot, el asistente virtual de la UNCAUS.
    Responde la pregunta del usuario basándote EXCLUSIVAMENTE en el siguiente contexto.

    CONTEXTO RECUPERADO DE LA BASE DE DATOS:
    "{contexto}"

    ---
    INSTRUCCIONES:
    1. Usa la información del CONTEXTO para responder.
    2. Si la respuesta NO está en el contexto, di textualmente: "Lo siento, no tengo información sobre ese tema específico en mi base de conocimientos de Alumnado."
    3. Sé amable, breve y usa emojis.
    """ 

    # 3. Generamos la respuesta con Gemini
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(prompt)
        respuesta_final = response.text
    except Exception as e:
        respuesta_final = "Lo siento, hubo un error técnico al procesar tu solicitud."
        print(f"❌ Error Gemini: {e}")

    # 4. Guardamos la conversación (Sin bloquear si falla)
    try:
        supabase.table("chat_logs").insert({
            "session_id": query.session_id,
            "user_input": query.pregunta,
            "bot_response": respuesta_final
        }).execute()
    except Exception as e:
        print(f"⚠️ No se pudo guardar el log (pero el bot respondió bien): {e}")

    return {"respuesta": respuesta_final}

# Endpoint de prueba para saber si el servidor está vivo
@app.get("/")
def home():
    return {"status": "UniBot está vivo y funcionando 🤖"}