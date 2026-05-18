import os
import sys
import json
import csv
import base64
import fitz
import chromadb
import re
from tqdm import tqdm
from openai import OpenAI

APP_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DB_PATH = os.path.join(APP_DIR, "chroma_db")
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
EXTRACTED_TEXT_DIR = os.path.join(APP_DIR, "extracted_text")
OCR_PAGES_DIR = os.path.join(APP_DIR, "ocr_pages")
COLLECTION_NAME = "pdf_rag_local"

WEAK_PAGE_WORD_THRESHOLD = 50


def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("No existe config.json junto al .exe o script.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


config = load_config()
client_ai = OpenAI(api_key=config["OPENAI_API_KEY"])

EMBED_MODEL = config.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = config.get("CHAT_MODEL", "gpt-4.1-mini")
OCR_MODEL = config.get("OCR_MODEL", "gpt-4.1")

def clean_response(text):

    text=re.sub(r'\*\*','',text)

    text=re.sub(
        r'Fuente:.*',
        '',
        text,
        flags=re.IGNORECASE
    )

    text=re.sub(
        r'Fuentes:.*',
        '',
        text,
        flags=re.IGNORECASE
    )

    return text.strip()

def get_collection():
    client_db = chromadb.PersistentClient(path=DB_PATH)
    return client_db.get_or_create_collection(name=COLLECTION_NAME)


def clean_text(text):
    return text.replace("\x00", "").strip()


def page_is_weak(text):
    return len(text.split()) < WEAK_PAGE_WORD_THRESHOLD


def render_page_to_image(pdf_path, page_index, zoom=2):
    os.makedirs(OCR_PAGES_DIR, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_index]

    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")
    image_path = os.path.join(OCR_PAGES_DIR, f"{base_name}_page_{page_index + 1}.png")

    pix.save(image_path)
    return image_path


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_page_with_gpt(image_path):
    image_b64 = image_to_base64(image_path)

    prompt = """
Extrae TODO el texto visible de esta página de PDF.

Reglas:
- No resumas.
- No interpretes.
- No inventes.
- Conserva títulos, subtítulos, numerales, tablas, montos, fechas, nombres propios y notas.
- Si hay tablas, conviértelas a texto estructurado manteniendo filas y columnas de la mejor manera posible.
- Si algo no es legible, escribe [ilegible].
- Devuelve solo el texto extraído.
"""

    response = client_ai.responses.create(
        model=OCR_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}"
                    }
                ]
            }
        ]
    )

    return response.output_text.strip()


def extract_text_from_pdf(pdf_path, force_ocr=False):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        normal_text = clean_text(page.get_text("text"))

        use_ocr = force_ocr or page_is_weak(normal_text)

        if use_ocr:
            try:
                image_path = render_page_to_image(pdf_path, i)
                ocr_text = clean_text(ocr_page_with_gpt(image_path))

                if ocr_text:
                    final_text = ocr_text
                    method = "OCR_GPT"
                else:
                    final_text = normal_text
                    method = "TEXT_WEAK_NO_OCR_RESULT"

            except Exception as e:
                final_text = normal_text
                method = f"OCR_ERROR: {str(e)}"
        else:
            final_text = normal_text
            method = "TEXT"

        if final_text.strip():
            pages.append({
                "page": i + 1,
                "text": final_text,
                "method": method,
                "words": len(final_text.split()),
                "chars": len(final_text)
            })

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

        for item in pages:
            f.write(f"\n\n=== PÁGINA {item['page']} | MÉTODO: {item['method']} | PALABRAS: {item['words']} ===\n\n")
            f.write(item["text"])

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


def index_folder(folder_path=None, force_ocr=False):
    if folder_path is None:
        folder_path = APP_DIR

    collection = get_collection()
    pdf_files = []

    for root, _, files in os.walk(folder_path):
        if "chroma_db" in root or "extracted_text" in root or "ocr_pages" in root:
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
        print(f"\nProcesando: {os.path.basename(pdf_path)}")

        pages = extract_text_from_pdf(pdf_path, force_ocr=force_ocr)

        if not pages:
            print(f"Advertencia: no se extrajo texto de {os.path.basename(pdf_path)}.")
            continue

        text_path = save_extracted_text(pdf_path, pages)
        print(f"Texto extraído guardado en: {text_path}")

        for item in pages:
            page_num = item["page"]
            text = item["text"]
            method = item["method"]

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
                        "page": page_num,
                        "method": method
                    }]
                )

                total_chunks += 1

    print(f"\nIndexación terminada. Chunks guardados: {total_chunks}")
    print(f"Base local creada en: {DB_PATH}")
    print(f"Textos completos guardados en: {EXTRACTED_TEXT_DIR}")
    print(f"Imágenes OCR guardadas en: {OCR_PAGES_DIR}")


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
        method = meta.get("method", "N/A")
        context += f"\n\n[Fuente: {meta['file']} | Página: {meta['page']} | Método: {method}]\n{doc}"

    prompt = f"""
Eres un analista experto en documentos técnicos, legales y de ingeniería.

REGLAS OBLIGATORIAS:

- Usa únicamente el contexto proporcionado.
- No inventes información.
- Si el contexto no es suficiente, dilo.
- NO uses markdown.
- NO uses **.
- NO uses listas markdown.
- NO escribas "Fuente:" dentro del texto.
- NO cites páginas dentro del cuerpo.
- Responde en texto limpio.
- Usa títulos simples.
- Usa numeración normal:
  1.
  2.
  3.

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
    cleaned=clean_response(
    response.choices[0].message.content
    )

    print(cleaned)

    print("\nFUENTES RECUPERADAS:")
print("\nFUENTES:\n")

seen=set()

    for meta in metadatas:
    
        source=f"{meta['file']} | Página {meta['page']}"
    
        if source not in seen:
    
            print(
                f"• {source}"
            )
    
            seen.add(source)


def audit_extracted_text():
    if not os.path.exists(EXTRACTED_TEXT_DIR):
        print("No existe la carpeta extracted_text. Primero ejecuta: PDF_RAG_LOCAL.exe index")
        return

    report_path = os.path.join(APP_DIR, "coverage_report.csv")
    rows = []

    txt_files = [
        f for f in os.listdir(EXTRACTED_TEXT_DIR)
        if f.lower().endswith("_full.txt")
    ]

    if not txt_files:
        print("No encontré archivos *_full.txt en extracted_text.")
        return

    for txt_file in txt_files:
        path = os.path.join(EXTRACTED_TEXT_DIR, txt_file)

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        page_markers = content.count("=== PÁGINA")
        total_chars = len(content)
        total_words = len(content.split())

        pages = content.split("=== PÁGINA")
        empty_pages = 0
        weak_pages = 0
        ocr_pages = content.count("MÉTODO: OCR_GPT")
        text_pages = content.count("MÉTODO: TEXT")

        for page in pages[1:]:
            page_text = page.strip()
            word_count = len(page_text.split())

            if word_count == 0:
                empty_pages += 1
            elif word_count < 50:
                weak_pages += 1

        if page_markers == 0:
            coverage_score = 0
        else:
            weak_ratio = (empty_pages + weak_pages) / page_markers
            coverage_score = max(0, round((1 - weak_ratio) * 100, 2))

        if coverage_score >= 95:
            status = "Excelente captura"
        elif coverage_score >= 75:
            status = "Buena captura con posibles pérdidas"
        elif coverage_score >= 40:
            status = "Captura débil: revisar PDF/OCR"
        else:
            status = "Muy baja captura: probable PDF escaneado"

        rows.append({
            "archivo_txt": txt_file,
            "paginas_detectadas": page_markers,
            "paginas_texto_normal": text_pages,
            "paginas_ocr_gpt": ocr_pages,
            "paginas_vacias": empty_pages,
            "paginas_debiles_menos_50_palabras": weak_pages,
            "total_caracteres": total_chars,
            "total_palabras": total_words,
            "score_captura_aprox": coverage_score,
            "diagnostico": status
        })

    with open(report_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "archivo_txt",
            "paginas_detectadas",
            "paginas_texto_normal",
            "paginas_ocr_gpt",
            "paginas_vacias",
            "paginas_debiles_menos_50_palabras",
            "total_caracteres",
            "total_palabras",
            "score_captura_aprox",
            "diagnostico"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Auditoría terminada.")
    print(f"Reporte creado en: {report_path}")

    for row in rows:
        print(f"\nArchivo: {row['archivo_txt']}")
        print(f"Páginas detectadas: {row['paginas_detectadas']}")
        print(f"Páginas con texto normal: {row['paginas_texto_normal']}")
        print(f"Páginas con OCR GPT: {row['paginas_ocr_gpt']}")
        print(f"Score captura aprox: {row['score_captura_aprox']}%")
        print(f"Diagnóstico: {row['diagnostico']}")


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

3) Indexar forzando OCR GPT en todas las páginas:
PDF_RAG_LOCAL.exe index-ocr

4) Preguntar una sola vez:
PDF_RAG_LOCAL.exe ask "¿Qué dice el documento sobre penalidades?"

5) Modo conversación:
PDF_RAG_LOCAL.exe chat

6) Auditar captura:
PDF_RAG_LOCAL.exe audit

Archivos necesarios junto al .exe:
- PDF_RAG_LOCAL.exe
- config.json

Carpetas creadas automáticamente:
- chroma_db/
- extracted_text/
- ocr_pages/
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit()

    command = sys.argv[1].lower()

    if command == "index":
        if len(sys.argv) >= 3:
            index_folder(sys.argv[2], force_ocr=False)
        else:
            index_folder(APP_DIR, force_ocr=False)

    elif command == "index-ocr":
        if len(sys.argv) >= 3:
            index_folder(sys.argv[2], force_ocr=True)
        else:
            index_folder(APP_DIR, force_ocr=True)

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Falta la pregunta.")
            sys.exit()

        ask_question(sys.argv[2])

    elif command == "chat":
        chat_mode()

    elif command == "audit":
        audit_extracted_text()

    else:
        show_help()
