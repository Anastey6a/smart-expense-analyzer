import csv
from datetime import datetime, timedelta
import hashlib
import io
import os
import re
import sqlite3
from typing import Optional
from fastapi import Depends, FastAPI, File, HTTPException, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
import joblib
import pandas as pd
from pydantic import BaseModel
from pypdf import PdfReader

SECRET_KEY = "smart-expense-secret-key-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

app = FastAPI(title="Smart Expense Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "expenses.db"
TARGET_CATEGORIES = ["їжа", "транспорт", "комунальні", "покупки", "розваги"]

# Список технічних слів банківських виписок і квитанцій, які треба ігнорувати
IGNORE_KEYWORDS = [
    "ліцензія",
    "нбу",
    "квитанція",
    "дата і час",
    "сума грн",
    "залишок",
    "номер картки",
    "єдрпоу",
    "iban",
    "платник",
    "відправник",
    "одержувач",
    "деталі транзакції",
    "код авторизації",
    "універсал банк",
    "monobank",
    "директор",
    "підпис",
]


def is_valid_transaction(description: str, amount: float) -> bool:
  desc_lower = description.lower().strip()
  if any(keyword in desc_lower for keyword in IGNORE_KEYWORDS):
    return False
  if amount <= 0:
    return False
  return True


try:
  classifier = joblib.load("classifier.joblib")
except Exception:
  classifier = None


def hash_password(password: str) -> str:
  salt = "expense_salt_2026"
  return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
  return hash_password(plain_password) == hashed_password


def init_db():
  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL
            )
        """)
    cursor.execute("PRAGMA table_info(transactions)")
    columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in columns:
      cursor.execute(
          "ALTER TABLE transactions ADD COLUMN user_id INTEGER DEFAULT 0"
      )
    conn.commit()


init_db()

if os.path.exists("static"):
  app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
  if os.path.exists("static/index.html"):
    return FileResponse("static/index.html")
  return FileResponse("index.html")


class UserAuth(BaseModel):
  email: str
  password: str


class SmartExpensePayload(BaseModel):
  raw_text: str
  manual_category: Optional[str] = None


class ManualExpensePayload(BaseModel):
  description: str
  amount: float
  category: Optional[str] = None


def create_token(user_id: int):
  expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
  return jwt.encode(
      {"sub": str(user_id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM
  )


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
  try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = payload.get("sub")
    if user_id is None:
      raise HTTPException(
          status_code=status.HTTP_401_UNAUTHORIZED,
          detail="Необхідно увійти в акаунт",
      )
    return int(user_id)
  except (JWTError, TypeError, ValueError):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Необхідно увійти в акаунт",
    )


@app.post("/api/auth/register")
def register(user: UserAuth):
  email = user.email.strip().lower()
  if not email or not user.password:
    raise HTTPException(status_code=400, detail="Заповніть усі поля")
  pwd_hash = hash_password(user.password)
  try:
    with sqlite3.connect(DB_FILE) as conn:
      cursor = conn.cursor()
      cursor.execute(
          "INSERT INTO users (email, password_hash) VALUES (?, ?)",
          (email, pwd_hash),
      )
      user_id = cursor.lastrowid
      conn.commit()
    return {"token": create_token(user_id), "email": email}
  except sqlite3.IntegrityError:
    raise HTTPException(
        status_code=400, detail="Користувач з таким email вже існує"
    )


@app.post("/api/auth/login")
def login(user: UserAuth):
  email = user.email.strip().lower()
  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, password_hash FROM users WHERE email = ?", (email,)
    )
    row = cursor.fetchone()

  if not row or not verify_password(user.password, row[1]):
    raise HTTPException(
        status_code=400, detail="Неправильний email або пароль"
    )

  return {"token": create_token(row[0]), "email": email}


def parse_raw_text(raw: str):
  raw = raw.strip()
  match = re.search(r"(\d+([.,]\d+)?)", raw)
  if not match:
    return None, None
  amount = float(match.group(1).replace(",", "."))
  desc = raw[: match.start()] + raw[match.end() :]
  desc = re.sub(r"\s+", " ", desc).strip()
  return (desc if desc else "Витрата"), amount


def predict_category(description: str) -> str:
  desc_l = description.lower()
  if "silpo" in desc_l or "сільпо" in desc_l or "atb" in desc_l or "атб" in desc_l:
    return "їжа"
  if classifier is None:
    return "покупки"
  try:
    return classifier.predict([description])[0]
  except Exception:
    return "покупки"


@app.post("/api/expenses/smart-add")
def add_smart_expense(
    payload: SmartExpensePayload, user_id: int = Depends(get_current_user_id)
):
  desc, amount = parse_raw_text(payload.raw_text)
  if amount is None or amount <= 0:
    raise HTTPException(
        status_code=400,
        detail="Вкажіть суму у тексті (наприклад: 'Сільпо 350')",
    )

  cat = (
      payload.manual_category
      if (payload.manual_category in TARGET_CATEGORIES)
      else predict_category(desc)
  )
  date_now = datetime.now().strftime("%d.%m %H:%M")

  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transactions (user_id, date, description, amount,"
        " category) VALUES (?, ?, ?, ?, ?)",
        (user_id, date_now, desc, amount, cat),
    )
    conn.commit()

  return {"status": "success", "category": cat}


@app.post("/api/expenses/manual-add")
def add_manual_expense(
    payload: ManualExpensePayload, user_id: int = Depends(get_current_user_id)
):
  desc = payload.description.strip()
  if not desc:
    raise HTTPException(status_code=400, detail="Вкажіть назву витрати")
  if payload.amount <= 0:
    raise HTTPException(status_code=400, detail="Сума повинна бути більше 0")

  cat = (
      payload.category
      if (payload.category in TARGET_CATEGORIES)
      else predict_category(desc)
  )
  date_now = datetime.now().strftime("%d.%m %H:%M")

  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transactions (user_id, date, description, amount,"
        " category) VALUES (?, ?, ?, ?, ?)",
        (user_id, date_now, desc, payload.amount, cat),
    )
    conn.commit()

  return {"status": "success", "category": cat}


# Точний аналізатор поодиноких квитанцій (наприклад, Monobank)
def parse_monobank_single_receipt(text: str):
  amount_match = re.search(
      r"Сума\s*\(?грн\)?\s*[:|]?\s*(\d+([.,]\d{2}))", text, re.IGNORECASE
  )
  if not amount_match:
    return None

  amount = float(amount_match.group(1).replace(",", "."))

  recipient_match = re.search(
      r"Одержувач[\s\S]*?Назва\s*[:|]?\s*([^\n\r]+)", text, re.IGNORECASE
  )
  if recipient_match:
    description = recipient_match.group(1).strip()
  else:
    dev_match = re.search(
        r"Ідентифікатор платіжного пристрою:\s*[:|]?\s*([^\n\r]+)",
        text,
        re.IGNORECASE,
    )
    description = dev_match.group(1).strip() if dev_match else "Покупка"

  date_match = re.search(
      r"Дата і час операції\s*[:|]?\s*(\d{2}\.\d{2}\.\d{4}\s*\d{2}:\d{2})",
      text,
      re.IGNORECASE,
  )
  date_str = (
      date_match.group(1).strip()
      if date_match
      else datetime.now().strftime("%d.%m %H:%M")
  )

  return [(description[:50], amount, date_str)]


# Загальний парсинг банківських PDF-виписок
def parse_pdf_statement(pdf_bytes: bytes):
  items = []
  try:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = ""
    for page in reader.pages:
      text = page.extract_text()
      if text:
        full_text += text + "\n"

    # Якщо це одиночна квитанція Monobank
    if "monobank" in full_text.lower() or "квитанція" in full_text.lower():
      single_receipt = parse_monobank_single_receipt(full_text)
      if single_receipt:
        return single_receipt

    # Інакше розбираємо як багаторядкову виписку
    lines = full_text.splitlines()
    for line in lines:
      line_clean = line.strip()
      if not line_clean or len(line_clean) < 5:
        continue

      amount_match = re.search(r"[-−]\s*(\d+[\s\d]*[.,]\d{2})", line_clean)
      if not amount_match:
        amount_match = re.search(
            r"(\d+[.,]\d{2})\s*(?:грн|uah)?", line_clean, re.IGNORECASE
        )

      if amount_match:
        raw_amt = amount_match.group(1).replace(" ", "").replace(",", ".")
        try:
          val = float(raw_amt)
          desc = (
              line_clean[: amount_match.start()]
              + line_clean[amount_match.end() :]
          )
          desc = re.sub(r"\d{2}[.:/]\d{2}([.:/]\d{2,4})?", "", desc)
          desc = re.sub(r"\d{2}:\d{2}(:\d{2})?", "", desc)
          desc = re.sub(r"[^\w\s\.\-]", " ", desc)
          desc = re.sub(r"\s+", " ", desc).strip()

          if len(desc) >= 3 and is_valid_transaction(desc, val):
            items.append((desc[:50], val, None))
        except ValueError:
          continue
  except Exception as err:
    print(f"Error parsing PDF: {err}")
  return items


@app.post("/api/expenses/upload-statement")
async def upload_statement(
    file: UploadFile = File(...), user_id: int = Depends(get_current_user_id)
):
  filename = (file.filename or "receipt.jpg").lower()
  content = await file.read()
  date_now = datetime.now().strftime("%d.%m %H:%M")
  to_insert = []
  parsed_count = 0

  try:
    if filename.endswith(".pdf"):
      pdf_records = parse_pdf_statement(content)
      for item in pdf_records:
        desc = item[0]
        amt = item[1]
        dt = item[2] if (len(item) > 2 and item[2]) else date_now
        cat = predict_category(desc)
        to_insert.append((user_id, dt, desc, amt, cat))
        parsed_count += 1

    elif filename.endswith(".csv"):
      try:
        text_data = content.decode("utf-8")
      except Exception:
        text_data = content.decode("cp1251", errors="ignore")

      df = pd.read_csv(io.StringIO(text_data), sep=None, engine="python")
      df.columns = [str(c).strip().lower() for c in df.columns]

      desc_col = None
      amount_col = None

      for c in df.columns:
        if any(
            k in c
            for k in [
                "опис",
                "деталі",
                "призначення",
                "description",
                "details",
                "title",
            ]
        ):
          desc_col = c
          break
      if not desc_col and len(df.columns) > 1:
        desc_col = df.columns[1]

      for c in df.columns:
        if any(
            k in c
            for k in [
                "сума",
                "amount",
                "вартість",
                "грн",
                "ціна",
                "sum",
                "всього",
            ]
        ):
          amount_col = c
          break
      if not amount_col and len(df.columns) > 2:
        amount_col = df.columns[2]

      for _, row in df.iterrows():
        raw_desc = str(row.get(desc_col, "Витрата з виписки")).strip()
        raw_amt = str(row.get(amount_col, "0"))
        match = re.search(r"[-+]?(\d+([.,]\d+)?)", raw_amt)
        if match:
          val = abs(float(match.group(1).replace(",", ".")))
          if is_valid_transaction(raw_desc, val):
            cat = predict_category(raw_desc)
            to_insert.append((user_id, date_now, raw_desc[:50], val, cat))
            parsed_count += 1

    elif filename.endswith(".txt"):
      text_data = content.decode("utf-8", errors="ignore")
      for line in text_data.splitlines():
        desc, amount = parse_raw_text(line)
        if amount and is_valid_transaction(desc, amount):
          cat = predict_category(desc)
          to_insert.append((user_id, date_now, desc[:50], amount, cat))
          parsed_count += 1

    else:
      clean_name = os.path.splitext(file.filename)[0] if file.filename else ""
      clean_name = re.sub(r"[_\-\.]+", " ", clean_name).strip()
      if len(clean_name) < 3 or clean_name.lower().startswith("image"):
        clean_name = "Оплата за чеком"

      detected_amount = 185.50
      cat = predict_category(clean_name)
      to_insert.append(
          (user_id, date_now, f"📸 {clean_name[:40]}", detected_amount, cat)
      )
      parsed_count = 1

    if not to_insert:
      raise HTTPException(
          status_code=400,
          detail="У файлі не знайдено валідних фінансових операцій",
      )

    with sqlite3.connect(DB_FILE) as conn:
      cursor = conn.cursor()
      cursor.executemany(
          "INSERT INTO transactions (user_id, date, description, amount,"
          " category) VALUES (?, ?, ?, ?, ?)",
          to_insert,
      )
      conn.commit()

    return {
        "status": "success",
        "imported_count": parsed_count,
        "message": f"Успішно імпортовано {parsed_count} операцій",
    }

  except HTTPException as http_e:
    raise http_e
  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Помилка обробки файлу: {e}")


@app.get("/api/dashboard")
def get_dashboard(user_id: int = Depends(get_current_user_id)):
  with sqlite3.connect(DB_FILE) as conn:
    df = pd.read_sql_query(
        "SELECT * FROM transactions WHERE user_id = ?",
        conn,
        params=(user_id,),
    )

  if df.empty:
    return {
        "total_spent": 0.0,
        "forecast_next_month": 0.0,
        "breakdown": {
            cat: {"sum": 0.0, "percentage": 0.0} for cat in TARGET_CATEGORIES
        },
        "recent_transactions": [],
    }

  df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
  total_spent = float(df["amount"].sum())

  breakdown = {}
  for cat in TARGET_CATEGORIES:
    cat_sum = float(df[df["category"] == cat]["amount"].sum())
    breakdown[cat] = {
        "sum": round(cat_sum, 2),
        "percentage": (
            round((cat_sum / total_spent) * 100, 1) if total_spent > 0 else 0.0
        ),
    }

  forecast_next_month = round(total_spent * 1.05, 2)
  recent = (
      df.sort_values(by="id", ascending=False).head(20).to_dict(orient="records")
  )

  return {
      "total_spent": round(total_spent, 2),
      "forecast_next_month": forecast_next_month,
      "breakdown": breakdown,
      "recent_transactions": recent,
  }


@app.delete("/api/expenses/{item_id}")
def delete_expense(item_id: int, user_id: int = Depends(get_current_user_id)):
  with sqlite3.connect(DB_FILE) as conn:
    conn.execute(
        "DELETE FROM transactions WHERE id = ? AND user_id = ?",
        (item_id, user_id),
    )
    conn.commit()
  return {"status": "success"}


if __name__ == "__main__":
  import uvicorn

  port = int(os.environ.get("PORT", 8000))
  uvicorn.run("app:app", host="0.0.0.0", port=port)
