import os
import base64
import anthropic
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SMARTRU_API_URL   = os.environ.get("SMARTRU_API_URL", "https://semdesperdicio.smartru.com.br/api")
ADMIN_API_KEY     = os.environ.get("ADMIN_API_KEY", "")

POSTGRES_HOST     = os.environ["POSTGRES_HOST"]
POSTGRES_PORT     = int(os.environ.get("POSTGRES_PORT", 5432))
POSTGRES_DB       = os.environ["POSTGRES_DB"]
POSTGRES_USER     = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI(
    title="SmartRU Menu Analyzer",
    description="Extrai pratos do cardápio usando Claude Vision",
    version="1.0.0",
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

# ─── Claude Vision ────────────────────────────────────
def analyze_menu_image(image_bytes: bytes, meal_type: str = "almoco") -> dict:
    """
    Envia a imagem do cardápio para o Claude Vision e extrai os pratos.
    Retorna um dicionário com os pratos organizados.
    """
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

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    # Remove blocos de código se existirem
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ─── Busca imagem da API SmartRU ──────────────────────
def fetch_menu_image(menu_id: int, meal_type: str = "lunch") -> bytes | None:
    try:
        r = requests.get(
            f"{SMARTRU_API_URL}/menu/image/{menu_id}/{meal_type}",
            headers={"Authorization": f"Bearer {ADMIN_API_KEY}"},
            timeout=10,
        )
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


# ─── Guarda no banco ──────────────────────────────────
def save_dishes_to_db(menu_id: int, dishes: dict):
    """Guarda os pratos extraídos no campo dishes da tabela menu."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE menu SET dishes = %s WHERE id = %s",
                (json.dumps(dishes), menu_id)
            )
        conn.commit()


# ─── Endpoints ────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartRU Menu Analyzer"}


class AnalyzeRequest(BaseModel):
    menu_id: int
    meal_type: str = "lunch"  # "lunch" ou "dinner"


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Analisa a imagem do cardápio e extrai os pratos.
    Guarda o resultado no banco de dados.
    """
    # 1. Busca a imagem na API SmartRU
    image_bytes = fetch_menu_image(req.menu_id, req.meal_type)
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Imagem do cardápio não encontrada.")

    # 2. Claude Vision analisa a imagem
    try:
        dishes = analyze_menu_image(image_bytes, req.meal_type)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao processar resposta do Claude Vision.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem: {str(e)}")

    if "erro" in dishes:
        raise HTTPException(status_code=400, detail=dishes["erro"])

    # 3. Guarda no banco (só funciona após o Iarley criar o campo dishes)
    db_saved = False
    try:
        save_dishes_to_db(req.menu_id, dishes)
        db_saved = True
    except Exception as e:
        # Não falha — retorna os dados mesmo sem guardar no banco
        print(f"Aviso: não foi possível guardar no banco: {e}")

    return {
        "menu_id":  req.menu_id,
        "meal_type": req.meal_type,
        "dishes":   dishes,
        "db_saved": db_saved,
        "message":  "Pratos extraídos com sucesso!" if db_saved else "Pratos extraídos mas não guardados no banco ainda — aguarda a migration.",
    }


@app.get("/menu/{menu_id}/dishes")
def get_dishes(menu_id: int):
    """Retorna os pratos já extraídos de um cardápio."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT dishes, uploaded_at FROM menu WHERE id = %s", (menu_id,))
                row = cur.fetchone()
                if not row or not row["dishes"]:
                    raise HTTPException(status_code=404, detail="Pratos não extraídos ainda para este cardápio.")
                return {"menu_id": menu_id, "dishes": row["dishes"], "uploaded_at": str(row["uploaded_at"])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/menu/current/dishes")
def get_current_dishes():
    """Retorna os pratos do cardápio atual."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, dishes, uploaded_at FROM menu WHERE dishes IS NOT NULL ORDER BY uploaded_at DESC LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Nenhum cardápio com pratos extraídos encontrado.")
                return {"menu_id": row["id"], "dishes": row["dishes"], "uploaded_at": str(row["uploaded_at"])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
