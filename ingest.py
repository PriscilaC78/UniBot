import os
import google.generativeai as genai
from supabase import create_client, Client
from langchain_community.document_loaders import PyPDFLoader
# --- ESTA ES LA LÍNEA CORREGIDA (con guion bajo) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configurar clientes
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

def run_ingest():
    print("🚀 Iniciando proceso de ingestión...")

    # A. LIMPIEZA DE DATOS ANTIGUOS
    print("🗑️  Borrando datos antiguos...")
    try:
        supabase.table("documents").delete().neq("id", 0).execute()
    except Exception as e:
        print(f"⚠️ Nota: No se pudo limpiar la tabla (quizás estaba vacía): {e}")

    # B. CARGAR EL PDF
    pdf_path = "faq.pdf" 
    if not os.path.exists(pdf_path):
        print(f"❌ Error: No encuentro el archivo '{pdf_path}'")
        return

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"📄 PDF cargado. Total de páginas: {len(docs)}")
    except Exception as e:
        print(f"❌ Error al leer el PDF: {e}")
        return

    # C. DIVIDIR EL TEXTO
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Texto dividido en {len(chunks)} fragmentos.")

    # D. GENERAR EMBEDDINGS Y SUBIR
    print("🧠 Generando vectores y subiendo a Supabase... (Ignora las advertencias de Google)")
    
    for i, chunk in enumerate(chunks):
        content = chunk.page_content
        
        try:
            # Generar embedding
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=content,
                task_type="retrieval_document"
            )
            embedding = response['embedding']

            # Preparar y subir
            data = {
                "content": content,
                "metadata": chunk.metadata,
                "embedding": embedding
            }
            supabase.table("documents").insert(data).execute()
            
            if i % 5 == 0:
                print(f"   ... Procesado fragmento {i+1}/{len(chunks)}")
                
        except Exception as e:
            print(f"⚠️ Error en el fragmento {i}: {e}")

    print("✅ ¡Listo! La base de datos de UniBot ha sido actualizada.")

if __name__ == "__main__":
    run_ingest()