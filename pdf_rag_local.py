import os
import sys
import json
import fitz
import chromadb
from tqdm import tqdm
from openai import OpenAI

APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DB_PATH = os.path.join(APP_DIR, "chroma_db")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
EXTRACTED_TEXT_DIR = os.path.join(APP_DIR, "extracted_text")
COLLECTION_NAME = "pdf_rag_local"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("No existe config.json junto al .exe o script.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


config = load_config()
client_ai = OpenAI(api_key=config["OPENAI_API_KEY"])
EMBED_MODEL = config.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = config.get("CHAT_MODEL", "gpt-4.1-mini")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append((i + 1, text))

    return pages


def save_extracted_text(pdf_path, pages):
    os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

    base_name = os.path.basename(pdf_path)
    safe_name = os.path.splitext(base_name)[0].replace(" ", "_")
    full_text_path = os.path.join(EXTRACTED_TEXT_DIR, f"{safe_name}_full.txt")

    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(f"ARCHIVO: {base_name}\n")
        f.write(f"RUTA: {pdf_path}\n")
        f.write("=" * 80 + "\n\n")

        for page_num, text in pages:
            f.write(f"\n\n=== PÁGINA {page_num} ===\n\n")
            f.write(text)

    return full_text_path


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def get_embedding(text):
    response = client_ai.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_collection():
    client_db = chromadb.PersistentClient(path=DB_PATH)
    return client_db.get_or_create_collection(name=COLLECTION_NAME)


def index_folder(folder_path=None):
    if folder_path is None:
        folder_path = APP_DIR

    collection = get_collection()
    pdf_files = []

    for root, _, files in os.walk(folder_path):
        if "chroma_db" in root or "extracted_text" in root:
            continue

        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        print("No se encontraron PDFs.")
        print(f"Carpeta revisada: {folder_path}")
        return

    print(f"Carpeta indexada: {folder_path}")
    print(f"PDFs encontrados: {len(pdf_files)}")

    total_chunks = 0

    for pdf_path in tqdm(pdf_files, desc="Indexando PDFs"):
        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            print(f"\nAdvertencia: no se extrajo texto de {os.path.basename(pdf_path)}.")
            print("Posible PDF escaneado. Necesitaría OCR.")
            continue

        text_path = save_extracted_text(pdf_path, pages)
        print(f"\nTexto extraído guardado en: {text_path}")

        for page_num, text in pages:
            chunks = chunk_text(text)

            for idx, chunk in enumerate(chunks):
                doc_id = f"{pdf_path}_{page_num}_{idx}"

                embedding = get_embedding(chunk)

                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "file": os.path.basename(pdf_path),
                        "path": pdf_path,
                        "page": page_num
                    }]
                )

                total_chunks += 1

    print(f"\nIndexación terminada. Chunks guardados: {total_chunks}")
    print(f"Base local creada en: {DB_PATH}")
    print(f"Textos completos guardados en: {EXTRACTED_TEXT_DIR}")


def ask_question(question, top_k=15):
    collection = get_collection()

    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        print("No encontré información relevante en los PDFs indexados.")
        return

    context = ""

    for doc, meta in zip(documents, metadatas):
        context += f"\n\n[Fuente: {meta['file']} | Página: {meta['page']}]\n{doc}"

    prompt = f"""
Eres un analista experto en documentos técnicos, legales y de ingeniería.

Responde usando SOLO el contexto proporcionado.
No inventes información.
Si el contexto no es suficiente, dilo claramente.
Incluye fuentes con archivo y página.
Sé preciso, ejecutivo y útil.

PREGUNTA:
{question}

CONTEXTO:
{context}

RESPUESTA:
"""

    response = client_ai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    print("\nRESPUESTA:\n")
    print(response.choices[0].message.content)

    print("\nFUENTES RECUPERADAS:")
    for meta in metadatas:
        print(f"- {meta['file']} | Página {meta['page']}")


def chat_mode():
    print("\nPDF_RAG_LOCAL - MODO CONVERSACIÓN")
    print("Escribe tu pregunta y presiona Enter.")
    print("Comandos:")
    print("- salir / exit / quit : cerrar")
    print("- clear               : limpiar pantalla\n")

    while True:
        question = input("Pregunta > ").strip()

        if question.lower() in ["salir", "exit", "quit"]:
            print("Cerrando chat.")
            break

        if question.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        if not question:
            continue

        ask_question(question)
        print("\n" + "=" * 80 + "\n")


def show_help():
    print("""
PDF_RAG_LOCAL - Chat local con PDFs usando OpenAI API

Uso:

1) Indexar PDFs en la misma carpeta del .exe:
PDF_RAG_LOCAL.exe index

2) Indexar una carpeta específica:
PDF_RAG_LOCAL.exe index "D:\\MIS_PDFS"

3) Preguntar una sola vez:
PDF_RAG_LOCAL.exe ask "¿Qué dice el documento sobre penalidades?"

4) Modo conversación:
PDF_RAG_LOCAL.exe chat

Archivos necesarios junto al .exe:
- PDF_RAG_LOCAL.exe
- config.json

Carpetas creadas automáticamente:
- chroma_db/
- extracted_text/
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit()

    command = sys.argv[1].lower()

    if command == "index":
        if len(sys.argv) >= 3:
            index_folder(sys.argv[2])
        else:
            index_folder(APP_DIR)

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Falta la pregunta.")
            sys.exit()

        ask_question(sys.argv[2])

    elif command == "chat":
        chat_mode()

    else:
        show_help()
