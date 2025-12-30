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

# Configuración de permisos para que la web pueda hablar con el bot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" significa que permite conexiones desde cualquier sitio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conexión a Servicios
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not supabase_url or not google_api_key:
    raise ValueError("❌ Faltan las variables de entorno en el archivo .env")

supabase: Client = create_client(supabase_url, supabase_key)
genai.configure(api_key=google_api_key)

# Modelo de datos para recibir la pregunta
class UserQuery(BaseModel):
    pregunta: str
    session_id: str = "anonimo" # Para identificar al usuario (opcional por ahora)

def buscar_contexto(pregunta_usuario: str):
    """
    1. Convierte la pregunta en vectores.
    2. Busca en Supabase los fragmentos más parecidos.
    """
    try:
        # Generamos el embedding de la pregunta
        result = genai.embed_content(
            model="models/text-embedding-004", 
            content=pregunta_usuario,
            task_type="retrieval_query"
        )
        query_vector = result['embedding']

        # Llamamos a la función 'match_documents' de Supabase
        # IMPORTANTE: Baja el threshold a 0.4 o 0.5 para que sea más flexible
        response = supabase.rpc("match_documents", {
            "query_embedding": query_vector,
            "match_threshold": 0.5, 
            "match_count": 5
        }).execute()
        
        # Unimos todos los fragmentos de texto encontrados en uno solo
        contexto = "\n\n".join([item['content'] for item in response.data])
        return contexto
        
    except Exception as e:
        print(f"Error buscando contexto: {e}")
        return ""

@app.post("/chat")
async def chat_endpoint(query: UserQuery):
    print(f"📩 Pregunta recibida: {query.pregunta}")

    # --- NUEVO: DETECTOR DE SALUDOS (Para que no falle nunca) ---
    saludos = ["hola", "buen dia", "buen día", "buenas", "que tal", "hello"]
    mensaje_usuario = query.pregunta.lower().strip()
    
    # Si el usuario solo dice "hola" (o algo parecido), respondemos directo
    if any(s in mensaje_usuario for s in saludos) and len(mensaje_usuario) < 20:
        return {"respuesta": "¡Hola! 👋 Soy UniBot, el asistente virtual de Alumnado UNCAUS. ¿En qué trámite, fecha o requisito puedo ayudarte hoy?"}
    # ------------------------------------------------------------
    
    # 1. Buscamos información relevante en la base de datos
    contexto = buscar_contexto(query.pregunta)
    
    # 2. Armamos el Prompt para la IA
    prompt = f"""
    Eres UniBot, el asistente virtual de la UNCAUS.
    Tu tarea es responder preguntas basándote en el siguiente contexto obtenido de la base de datos.

    CONTEXTO RECUPERADO:
    "{contexto}"

    ---
    INSTRUCCIONES:
    1. Si la pregunta es sobre trámites, fechas o la universidad, RESPONDE ÚNICAMENTE usando la información del "CONTEXTO RECUPERADO".
    2. Si la respuesta NO está en el contexto, di: "Lo siento, no tengo información sobre ese tema específico en mi base de conocimientos de Alumnado."
    3. Sé amable, conciso y utiliza emojis.
    """ 

    # 3. Generamos la respuesta con Gemini
    try:
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(prompt)
        respuesta_final = response.text
    except Exception as e:
        respuesta_final = "Lo siento, hubo un error al procesar tu solicitud con la IA."
        print(f"Error Gemini: {e}")

    # 4. Guardamos el log
    try:
        supabase.table("chat_logs").insert({
            "session_id": query.session_id,
            "user_input": query.pregunta,
            "bot_response": respuesta_final
        }).execute()
    except Exception as e:
        print(f"No se pudo guardar el log: {e}")

    return {"respuesta": respuesta_final}

@app.get("/")
def home():
    return {"status": "UniBot está vivo y funcionando 🤖"}