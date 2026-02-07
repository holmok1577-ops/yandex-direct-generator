from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import time
import csv
import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import re
import random
import uuid
from datetime import datetime
import base64

app = FastAPI()

# Статика и шаблоны
app.mount("/static", StaticFiles(directory="static"), name="static")
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
templates = Jinja2Templates(directory="templates")

# Флаг отладки ART: при False не сохраняем диагностические файлы в images/
DEBUG_ART = False

# При выключенной отладке очищаем ранее сохранённые debug-файлы (art_*.json/txt),
# не трогая реальные изображения
def _cleanup_art_debug_files() -> None:
    try:
        for name in os.listdir(IMAGES_DIR):
            if name.startswith("art_") and (name.endswith(".json") or name.endswith(".txt")):
                p = os.path.join(IMAGES_DIR, name)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    except Exception:
        pass

if not DEBUG_ART:
    _cleanup_art_debug_files()

# Настройки Yandex Cloud (используем API-ключ)
# Загружаем переменные окружения из .env
load_dotenv()

# Читаем значения из .env с именами, которые сейчас используются в файле .env
def _clean(v: str | None) -> str:
    if v is None:
        return ""
    return v.strip().strip('"').strip("'")

API_KEY = _clean(os.getenv("YOUR_API_KEY"))
FOLDER_ID = _clean(os.getenv("YOUR_FOLDER_ID"))

YANDEX_GPT_URL = _clean(os.getenv("YOUR_YANDEX_GPT_URL"))
YANDEX_ART_URL = _clean(os.getenv("YOUR_YANDEX_ART_URL"))

# Директория для CSV
CSV_DIR = "generated_csv"
os.makedirs(CSV_DIR, exist_ok=True)



def _heuristic_utp_audience(product: str) -> Dict[str, str]:
    p = product.lower()
    if any(k in p for k in ["ноутбук", "laptop", "ultrabook"]):
        return {
            "utp": ["Производительность и автономность", "Лёгкий и тонкий корпус", "Гарантия и быстрый сервис"],
            "target_audience": "Студенты, фрилансеры, офисные пользователи"
        }
    if any(k in p for k in ["смартфон", "phone", "iphone", "android"]):
        return {
            "utp": ["Камера и NFC", "Быстрая зарядка", "Официальная гарантия"],
            "target_audience": "Покупатели электроники и мобильных гаджетов"
        }
    if any(k in p for k in ["холодильник", "стиральная", "пылесос", "микроволновка"]):
        return {
            "utp": ["Энергоэффективность", "Бесшумная работа", "Доставка и установка"],
            "target_audience": "Семьи и владельцы квартир"
        }
    return {
        "utp": ["Выгодная цена", "Быстрая доставка", "Официальная гарантия"],
        "target_audience": "Потенциальные покупатели онлайн-магазинов"
    }


def get_utp_and_audience(product: str) -> Dict[str, str]:
    """Определяет УТП и ЦА через Yandex GPT с ретраями и эвристикой."""
    prompt = (
        f"Определи 3 ключевых УТП и основную целевую аудиторию для товара: '{product}'. "
        "Верни JSON ровно таким объектом: {\"utp\": [\"...\", \"...\", \"...\"], \"target_audience\": \"...\"}. Без лишнего текста."
    )
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {"temperature": 0.4, "maxTokens": "200"},
        "messages": [{"role": "user", "text": prompt}]
    }
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(3):
        try:
            response = requests.post(YANDEX_GPT_URL, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            try:
                content = response.json()["result"]["alternatives"][0]["message"]["text"]
                data = json.loads(content)
                utp = data.get("utp") or []
                audience = data.get("target_audience") or ""
                if isinstance(utp, list) and len(utp) >= 2 and isinstance(audience, str) and audience.strip():
                    return {"utp": utp[:3], "target_audience": audience.strip()}
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.0)

    # Эвристический фоллбек, чтобы не было "Не определено/Общая аудитория"
    return _heuristic_utp_audience(product)



def generate_ad_text(product: str, utp: List[str], audience: str, variant_idx: int = 1) -> Dict[str, str]:
    """Генерирует текст объявления."""
    # Перемешиваем УТП, чтобы варианты отличались
    utp_shuffled = utp[:]
    random.shuffle(utp_shuffled)
    utp_str = ", ".join(utp_shuffled)
    prompt = (
        f"Ты — копирайтер по контекстной рекламе. Создай ВАРИАНТ №{variant_idx} объявления для Яндекс Директ. "
        "Требования: заголовок ≤81 символ, текст ≤150 символов, без клише, без повторов фраз между вариантами. "
        f"Товар: {product}. Не изменяй написание товара — используй строку ТОВАР как есть. "
        f"УТП (в случайном порядке): {utp_str}. ЦА: {audience}. "
        "Стилистика: разговорная, конкретные выгоды, избегай повторов слов, разные начальные конструкции. "
        "Верни строго JSON: {headline, text} без доп. текста."
    )
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {"temperature": 0.9, "maxTokens": "300"},
        "messages": [{"role": "user", "text": prompt}]
    }
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(YANDEX_GPT_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    try:
        content = response.json()["result"]["alternatives"][0]["message"]["text"]
        return json.loads(content)
    except (json.JSONDecodeError, KeyError):
        # Локальный фоллбек с вариативностью
        starts = [
            "Новый взгляд на", "Пора выбрать", "Оцените", "Успейте на", "Выгодно:"
        ]
        benefits = [
            "доставка за 1–2 дня", "официальная гарантия", "скидки и бонусы",
            "рассрочка без переплат", "выгодная цена сегодня"
        ]
        h = f"{random.choice(starts)} {product} — {random.choice(['купите сейчас', 'закажите онлайн', 'лучший выбор'])}!"
        # Не вставляем аудиторию в текст; используем её только в промпте модели
        t = f"{random.choice(utp) if utp else 'Преимущества'} · {random.choice(benefits)}."
        return {"headline": h[:81], "text": t[:150]}



def generate_image_prompt(ad_data: Dict[str, str]) -> str:
    """Создаёт промт для изображения."""
    prompt = (
        "Создай промт для Yandex ART (50 слов) на основе объявления. "
        f"Заголовок: {ad_data['headline']}. Текст: {ad_data['text']}. "
        "Стиль: реалистичный, цвета яркие, композиция чёткая."
    )
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {"temperature": 0.5, "maxTokens": "200"},
        "messages": [{"role": "user", "text": prompt}]
    }
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(YANDEX_GPT_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        try:
            return response.json()["result"]["alternatives"][0]["message"]["text"].strip()
        except (json.JSONDecodeError, KeyError):
            return f"Промо-изображение по теме: {ad_data.get('headline','товар')}"
    except requests.exceptions.RequestException:
        return f"Промо-изображение по теме: {ad_data.get('headline','товар')}"



def generate_image(prompt: str) -> str:
    """Генерирует изображение и возвращает URL."""
    # Пытаемся отправить совместимый с ART формат сообщений
    payload = {
        "modelUri": f"art://{FOLDER_ID}/yandex-art/latest",
        "generationOptions": {
            "mimeType": "image/jpeg",
            "size": {"width": 1280, "height": 720}
        },
        "messages": [
            {"text": prompt}
        ]
    }
    headers = {
        "Authorization": f"Api-Key {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(YANDEX_ART_URL, json=payload, headers=headers, timeout=20)
        # Если код не 200 — опционально логируем тело ошибки и выходим
        if response.status_code != 200:
            if DEBUG_ART:
                try:
                    err_path = os.path.join(IMAGES_DIR, f"art_error_init_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt")
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(
                            "status="+str(response.status_code)+"\n"+
                            "headers="+str(dict(response.headers))+"\n"+
                            "payload=\n"+json.dumps(payload, ensure_ascii=False)+"\n"+
                            "body=\n"+response.text
                        )
                except Exception:
                    pass
            return ""
        init_json = response.json()
        # Опциональный дамп инициализации
        if DEBUG_ART:
            try:
                with open(os.path.join(IMAGES_DIR, f"art_init_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
                    json.dump(init_json, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        operation_id = init_json.get("id")
        if not operation_id:
            # опционально сохраним как ошибку, если нет id
            if DEBUG_ART:
                try:
                    with open(os.path.join(IMAGES_DIR, f"art_error_noid_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
                        json.dump(init_json, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            return ""
        
        max_checks = 30
        for _ in range(max_checks):
            status_response = requests.get(
                f"https://llm.api.cloud.yandex.net/operations/{operation_id}",
                headers=headers,
                timeout=10
            )
            status_response.raise_for_status()
            data = status_response.json()
            # Опциональные дампы статусов и печать ключей
            if DEBUG_ART:
                try:
                    if data.get("done"):
                        with open(os.path.join(IMAGES_DIR, f"art_done_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    elif _ % 5 == 0:
                        with open(os.path.join(IMAGES_DIR, f"art_tick_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                try:
                    print("ART status keys:", list(data.keys()))
                except Exception:
                    pass
            if data.get("done"):
                resp = data.get("response", {})
                # Вариант 1: прямая ссылка в поле image / images / resources[0].url
                url = resp.get("image") or (resp.get("images", [None]) or [None])[0]
                if not url:
                    resources = resp.get("resources")
                    if isinstance(resources, list) and resources:
                        url = resources[0].get("url")
                # Вариант 2: внутри alternatives
                if not url:
                    alts = resp.get("alternatives") or []
                    if alts and isinstance(alts, list):
                        cand = alts[0]
                        url = cand.get("imageUrl") or cand.get("url")
                        # Иногда base64 может лежать в data/base64
                        b64 = cand.get("base64") or cand.get("data")
                        if not url and b64:
                            try:
                                raw = base64.b64decode(b64)
                                fname = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
                                abs_path = os.path.join(IMAGES_DIR, fname)
                                with open(abs_path, "wb") as f:
                                    f.write(raw)
                                return f"/images/{fname}"
                            except Exception:
                                pass
                # Вариант 3: base64 прямо в response
                if not url:
                    b64 = resp.get("image_base64") or resp.get("base64") or resp.get("data")
                    if b64:
                        try:
                            raw = base64.b64decode(b64)
                            fname = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
                            abs_path = os.path.join(IMAGES_DIR, fname)
                            with open(abs_path, "wb") as f:
                                f.write(raw)
                            return f"/images/{fname}"
                        except Exception:
                            pass

                # Если что-то попало в url, но это не ссылка, а, вероятно, base64 — попробуем декодировать
                if url:
                    is_probably_base64 = False
                    try:
                        if isinstance(url, str) and not (url.startswith("http://") or url.startswith("https://") or url.startswith("/images/") or url.startswith("data:")):
                            # Простая эвристика: очень длинная строка без префикса URL
                            is_probably_base64 = len(url) > 100
                    except Exception:
                        is_probably_base64 = False

                    if is_probably_base64:
                        try:
                            raw = base64.b64decode(url)
                            fname = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
                            abs_path = os.path.join(IMAGES_DIR, fname)
                            with open(abs_path, "wb") as f:
                                f.write(raw)
                            return f"/images/{fname}"
                        except Exception:
                            pass

                    return url
                return ""
            time.sleep(1.5)
        return ""
    except requests.exceptions.RequestException as e:
        if DEBUG_ART:
            try:
                with open(os.path.join(IMAGES_DIR, f"art_error_exc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"), "w", encoding="utf-8") as f:
                    f.write(str(e))
            except Exception:
                pass
        return ""


def save_image_from_url(url: str) -> str:
    """Скачивает изображение по URL и сохраняет локально, возвращает публичный путь /images/.. либо пустую строку."""
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        ext = "jpg"
        ct = r.headers.get("Content-Type", "")
        if "png" in ct:
            ext = "png"
        elif "jpeg" in ct or "jpg" in ct:
            ext = "jpg"
        elif "webp" in ct:
            ext = "webp"
        fname = f"img_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
        abs_path = os.path.join(IMAGES_DIR, fname)
        with open(abs_path, "wb") as f:
            f.write(r.content)
        return f"/images/{fname}"
    except requests.exceptions.RequestException:
        return ""



def save_ad_to_csv(ad_data: Dict[str, str], image_url: str, product: str, idx: int) -> str:
    """Сохраняет объявление в CSV."""
    # Санитизация имени файла под Windows: заменяем недопустимые символы
    safe_product = re.sub(r"[^\w\-]+", "_", product, flags=re.U)
    safe_product = safe_product.strip("._")[:60] or "product"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    filename = os.path.join(CSV_DIR, f"ad_{safe_product}_{idx}_{ts}_{uid}.csv")
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["headline", "text", "image_url", "landing_page"])
        writer.writeheader()
        writer.writerow({
            "headline": ad_data["headline"],
            "text": ad_data["text"],
            "image_url": image_url,
            "landing_page": ad_data["landing_page"]
        })
    return filename



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    product: str = Form(...),
    landing_page: str = Form(...)
):
    # Шаг 1: Определяем УТП и ЦА
    utp_audience = get_utp_and_audience(product)
    utp = utp_audience.get("utp", ["Не определено"])
    audience = utp_audience.get("target_audience", "Общая аудитория")

    # Шаг 2–4: Генерируем 3 разнообразных варианта с дедупликацией
    ads = []
    seen = set()
    for i in range(3):
        # Несколько попыток, чтобы получить уникальный вариант
        for attempt in range(3):
            try:
                ad_text = generate_ad_text(product, utp, audience, variant_idx=i + 1)
                key = (ad_text.get("headline", "").strip().lower(), ad_text.get("text", "").strip().lower())
                if key in seen:
                    continue
                image_prompt = generate_image_prompt(ad_text)
                image_val = generate_image(image_prompt)
                # Если generate_image вернул локальный путь /images/..., используем его напрямую
                if image_val.startswith("/images/"):
                    image_public = image_val
                else:
                    image_public = save_image_from_url(image_val)
                csv_file = save_ad_to_csv(
                    {**ad_text, "landing_page": landing_page},
                    image_public or image_val,
                    product,
                    i + 1
                )
                ads.append({
                    "headline": ad_text["headline"],
                    "text": ad_text["text"],
                    "image_url": image_public or image_val,
                    "csv_file": csv_file,
                    "landing_page": landing_page
                })
                seen.add(key)
                break
            except Exception:
                continue
    if not ads:
        return templates.TemplateResponse("index.html", {"request": request, "ads": [], "error": "Не удалось сгенерировать объявления. Проверьте доступ к Yandex Cloud и повторите попытку."})
    return templates.TemplateResponse("index.html", {"request": request, "ads": ads})


@app.post("/generate_one")
async def generate_one(
    product: str = Form(...),
    landing_page: str = Form(...),
    variant: int = Form(1)
):
    # УТП и ЦА для конкретного продукта
    utp_audience = get_utp_and_audience(product)
    utp = utp_audience.get("utp", ["Не определено"])
    audience = utp_audience.get("target_audience", "Общая аудитория")

    # Формируем один вариант
    try:
        ad_text = generate_ad_text(product, utp, audience, variant_idx=variant)
        image_prompt = generate_image_prompt(ad_text)
        image_val = generate_image(image_prompt)
        if image_val.startswith("/images/"):
            image_public = image_val
        else:
            image_public = save_image_from_url(image_val)
        csv_file = save_ad_to_csv(
            {**ad_text, "landing_page": landing_page},
            image_public or image_val,
            product,
            variant
        )
        return JSONResponse({
            "ok": True,
            "headline": ad_text.get("headline", ""),
            "text": ad_text.get("text", ""),
            "image_url": image_public or image_val,
            "csv_file": csv_file,
            "landing_page": landing_page
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": "Не удалось сгенерировать объявление."}, status_code=500)

@app.get("/download/{file_path:path}")
async def download_csv(file_path: str):
    # Безопасность: разрешаем скачивать только из каталога CSV_DIR
    normalized = os.path.normpath(file_path)
    if not normalized.startswith(CSV_DIR):
        # Если в шаблоне передан полный путь, приведём его к относительному от CSV_DIR
        filename = os.path.basename(normalized)
        normalized = os.path.join(CSV_DIR, filename)

    abs_path = os.path.abspath(normalized)
    if not abs_path.startswith(os.path.abspath(CSV_DIR)) or not os.path.exists(abs_path):
        return HTMLResponse(status_code=404, content="Файл не найден")
    return FileResponse(abs_path, media_type="text/csv", filename=os.path.basename(abs_path))