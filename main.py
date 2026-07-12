import os
import base64
import json
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ─── Config ───────────────────────────────────────────
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
SMARTRU_API_URL   = os.environ.get("SMARTRU_API_URL", "https://semdesperdicio.smartru.com.br/api")
ADMIN_API_KEY     = os.environ.get("ADMIN_API_KEY", "")

POSTGRES_HOST     = os.environ.get("POSTGRES_HOST", "")
POSTGRES_PORT     = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB       = os.environ.get("POSTGRES_DB", "")
POSTGRES_USER     = os.environ.get("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

app = FastAPI(
    title="SmartRU Menu Analyzer",
    description="Extrai pratos do cardápio usando Google Gemini Vision",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://semdesperdicio.smartru.com.br", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Banco de dados ───────────────────────────────────
def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor,
    )

# ─── Gemini Vision ────────────────────────────────────
def analyze_menu_image(image_bytes: bytes, meal_type: str = "lunch") -> dict:
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = f"""Analisa esta imagem do cardápio do Restaurante Universitário.

Extrai TODOS os pratos listados no cardápio e organiza em JSON.

Retorna APENAS o JSON, sem texto adicional, neste formato:
{{
  "meal_type": "{meal_type}",
  "dishes": {{
    "prato_principal": ["prato1", "prato2"],
    "guarnicao": ["item1", "item2"],
    "sobremesa": ["item1"],
    "salada": ["item1", "item2"],
    "suco": ["item1"],
    "outros": ["item1"]
  }},
  "tipos_refeicao": {{
    "essencial": ["prato1"],
    "leve_sabor": ["prato1"],
    "select": ["prato1"],
    "vegetariano": ["prato1"]
  }},
  "observacoes": "qualquer informação adicional relevante"
}}

Se não conseguires ler alguma parte da imagem, coloca null nesse campo.
Se a imagem não for um cardápio, retorna {{"erro": "Imagem não é um cardápio"}}.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ]
    }

    r = requests.post(GEMINI_URL, json=payload, timeout=30)

    if r.status_code != 200:
        raise Exception(f"Gemini API erro {r.status_code}: {r.text[:200]}")

    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ─── Guarda no banco ──────────────────────────────────
def save_dishes_to_db(menu_id: int, dishes: dict):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE menu SET dishes = %s WHERE id = %s",
                (json.dumps(dishes), menu_id)
            )
        conn.commit()


# ─── Endpoints ────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "SmartRU Menu Analyzer", "runtime": "Gemini Vision", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartRU Menu Analyzer", "runtime": "Gemini Vision"}


@app.post("/analyze/upload")
async def analyze_upload(
    menu_id: int,
    meal_type: str = "lunch",
    file: UploadFile = File(...)
):
    """
    Recebe a imagem diretamente por upload e analisa com Gemini Vision.
    """
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Ficheiro vazio.")

    try:
        dishes = analyze_menu_image(image_bytes, meal_type)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao processar resposta do Gemini Vision.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem: {str(e)}")

    if "erro" in dishes:
        raise HTTPException(status_code=400, detail=dishes["erro"])

    db_saved = False
    try:
        save_dishes_to_db(menu_id, dishes)
        db_saved = True
    except Exception as e:
        print(f"Aviso: não foi possível guardar no banco: {e}")

    return {
        "menu_id":   menu_id,
        "meal_type": meal_type,
        "dishes":    dishes,
        "db_saved":  db_saved,
        "message":   "Pratos extraídos com sucesso!" if db_saved else "Pratos extraídos. Aguarda a migration para guardar no banco.",
    }


class AnalyzeBase64Request(BaseModel):
    menu_id: int
    meal_type: str = "lunch"
    image_base64: str


@app.post("/analyze/base64")
def analyze_base64(req: AnalyzeBase64Request):
    """Recebe a imagem em base64 e analisa com Gemini Vision."""
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido.")

    try:
        dishes = analyze_menu_image(image_bytes, req.meal_type)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao processar resposta do Gemini Vision.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem: {str(e)}")

    if "erro" in dishes:
        raise HTTPException(status_code=400, detail=dishes["erro"])

    db_saved = False
    try:
        save_dishes_to_db(req.menu_id, dishes)
        db_saved = True
    except Exception as e:
        print(f"Aviso: não foi possível guardar no banco: {e}")

    return {
        "menu_id":   req.menu_id,
        "meal_type": req.meal_type,
        "dishes":    dishes,
        "db_saved":  db_saved,
        "message":   "Pratos extraídos com sucesso!" if db_saved else "Pratos extraídos. Aguarda a migration para guardar no banco.",
    }


@app.get("/menu/{menu_id}/dishes")
def get_dishes(menu_id: int):
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT dishes, uploaded_at FROM menu WHERE id = %s", (menu_id,))
                row = cur.fetchone()
                if not row or not row["dishes"]:
                    raise HTTPException(status_code=404, detail="Pratos não extraídos ainda.")
                return {"menu_id": menu_id, "dishes": row["dishes"], "uploaded_at": str(row["uploaded_at"])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
