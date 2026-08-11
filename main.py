import os
import base64
import json
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────
CF_API_TOKEN      = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CF_ACCOUNT_ID     = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
SMARTRU_API_URL   = os.environ.get("SMARTRU_API_URL", "https://semdesperdicio.smartru.com.br/api")
ADMIN_API_KEY     = os.environ.get("ADMIN_API_KEY", "")

# Modelo de visão gratuito do Cloudflare
CF_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"
CF_URL   = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{CF_MODEL}"

app = FastAPI(
    title="SmartRU Menu Analyzer",
    description="Extrai pratos do cardápio usando Cloudflare Workers AI",
    version="5.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://semdesperdicio.smartru.com.br", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Adaptação de formato ──────────────────────────────
def to_dishes_list(analysis: dict, meal_type: str) -> list[dict]:
    """
    O backend principal espera `dishes` como list[dict[str, Any]]
    (ver MenuDishesRequest), mas o Cloudflare Workers AI devolve uma
    estrutura aninhada por categoria. Esta função achata isso numa
    lista de pratos, cada um com sua categoria e, se houver, os
    tipos de refeição (essencial, leve_sabor, select, vegetariano)
    aos quais pertence.
    """
    dishes_by_categoria = analysis.get("dishes", {}) or {}
    tipos_por_prato = {}
    for tipo, pratos in (analysis.get("tipos_refeicao", {}) or {}).items():
        for prato in pratos:
            tipos_por_prato.setdefault(prato, []).append(tipo)

    items = []
    for categoria, pratos in dishes_by_categoria.items():
        for prato in pratos:
            items.append({
                "nome": prato,
                "categoria": categoria,
                "meal_type": meal_type,
                "tipos_refeicao": tipos_por_prato.get(prato, []),
                "observacoes": analysis.get("observacoes", ""),
            })
    return items


# ─── Cliente do backend principal (RU Sem Desperdício) ─
def save_dishes_via_api(menu_id: int, analysis: dict, meal_type: str) -> bool:
    """Salva os pratos extraídos chamando o backend principal via API,
    em vez de conectar direto no Postgres."""
    try:
        dishes_list = to_dishes_list(analysis, meal_type)
        r = requests.post(
            f"{SMARTRU_API_URL}/menu/{menu_id}/dishes",
            headers={"Authorization": f"Bearer {ADMIN_API_KEY}"},
            json={"dishes": dishes_list},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"Aviso: não foi possível salvar via API: {e}")
        return False


def get_dishes_via_api(menu_id: int) -> dict:
    """Busca os pratos já salvos chamando o backend principal via API."""
    r = requests.get(
        f"{SMARTRU_API_URL}/menu/{menu_id}/dishes",
        headers={"Authorization": f"Bearer {ADMIN_API_KEY}"},
        timeout=15,
    )
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="Pratos não extraídos ainda.")
    r.raise_for_status()
    return r.json()


# ─── Cloudflare Workers AI Vision ─────────────────────
def analyze_menu_image(image_bytes: bytes, meal_type: str = "lunch") -> dict:
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = f"""Analisa esta imagem do cardápio do Restaurante Universitário.

Extrai TODOS os pratos listados e retorna APENAS JSON neste formato:
{{
  "meal_type": "{meal_type}",
  "dishes": {{
    "prato_principal": ["prato1", "prato2"],
    "guarnicao": ["item1", "item2"],
    "sobremesa": ["item1"],
    "salada": ["item1"],
    "suco": ["item1"],
    "outros": ["item1"]
  }},
  "tipos_refeicao": {{
    "essencial": ["prato1"],
    "leve_sabor": ["prato1"],
    "select": ["prato1"],
    "vegetariano": ["prato1"]
  }},
  "observacoes": "informações adicionais"
}}

Retorna APENAS o JSON, sem texto adicional.
Se não for cardápio, retorna {{"erro": "Imagem não é um cardápio"}}.
"""

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 1000,
    }

    r = requests.post(
        CF_URL,
        headers={
            "Authorization": f"Bearer {CF_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if r.status_code != 200:
        raise Exception(f"Cloudflare AI erro {r.status_code}: {r.text[:300]}")

    result = r.json()
    print(f"Cloudflare raw response: {json.dumps(result)[:500]}")

    if not result.get("result"):
        raise Exception(f"Resposta vazia do Cloudflare AI: {result}")

    r_data = result["result"]
    response = r_data.get("response", "")

    # Se a resposta já é um dicionário (JSON parseado pelo Cloudflare), usa directamente
    if isinstance(response, dict):
        return response

    # Se é string, tenta fazer parse
    if isinstance(response, str):
        text = response.replace("```json", "").replace("```", "").strip()
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            text = json_match.group()
        return json.loads(text)

    raise Exception(f"Formato de resposta inesperado: {type(response)}")


# ─── Endpoints ────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "SmartRU Menu Analyzer", "runtime": "Cloudflare Workers AI", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "SmartRU Menu Analyzer", "runtime": "Cloudflare Workers AI"}


@app.post("/analyze/upload")
async def analyze_upload(
    menu_id: int,
    meal_type: str = "lunch",
    file: UploadFile = File(...)
):
    """Recebe imagem por upload e analisa com Cloudflare Workers AI."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Ficheiro vazio.")

    try:
        analysis = analyze_menu_image(image_bytes, meal_type)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao processar resposta do modelo.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem: {str(e)}")

    if "erro" in analysis:
        raise HTTPException(status_code=400, detail=analysis["erro"])

    db_saved = save_dishes_via_api(menu_id, analysis, meal_type)

    return {
        "menu_id":   menu_id,
        "meal_type": meal_type,
        "dishes":    analysis.get("dishes", {}),
        "db_saved":  db_saved,
        "message":   "Pratos extraídos com sucesso!" if db_saved else "Pratos extraídos, mas houve falha ao salvar via API.",
    }


class AnalyzeBase64Request(BaseModel):
    menu_id: int
    meal_type: str = "lunch"
    image_base64: str


@app.post("/analyze/base64")
def analyze_base64(req: AnalyzeBase64Request):
    """Recebe imagem em base64 e analisa com Cloudflare Workers AI."""
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido.")

    try:
        analysis = analyze_menu_image(image_bytes, req.meal_type)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Erro ao processar resposta do modelo.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem: {str(e)}")

    if "erro" in analysis:
        raise HTTPException(status_code=400, detail=analysis["erro"])

    db_saved = save_dishes_via_api(req.menu_id, analysis, req.meal_type)

    return {
        "menu_id":   req.menu_id,
        "meal_type": req.meal_type,
        "dishes":    analysis.get("dishes", {}),
        "db_saved":  db_saved,
        "message":   "Pratos extraídos com sucesso!" if db_saved else "Pratos extraídos, mas houve falha ao salvar via API.",
    }


@app.get("/menu/{menu_id}/dishes")
def get_dishes(menu_id: int):
    """Repassa a consulta pro backend principal em vez de ler direto do Postgres."""
    return get_dishes_via_api(menu_id)
