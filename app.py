from datetime import datetime, timedelta
import hashlib
import io
import os
import re
import sqlite3
from fastapi import Depends, FastAPI, File, HTTPException, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
import joblib
import pandas as pd
from pydantic import BaseModel

SECRET_KEY = "super-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

app = FastAPI(title="Smart Expense Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "expenses.db"
TARGET_CATEGORIES = ["їжа", "транспорт", "комунальні", "покупки", "розваги"]

try:
  classifier = joblib.load("classifier.joblib")
except Exception:
  classifier = None


# Надійне хешування без зовнішніх несумісних бібліотек
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
                category TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
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
  manual_category: str | None = None


def create_token(user_id: int):
  expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
  return jwt.encode(
      {"sub": str(user_id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM
  )


def get_current_user_id(token: str = Depends(oauth2_scheme)) -> int:
  try:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    user_id = int(payload.get("sub"))
    return user_id
  except (JWTError, TypeError, ValueError):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Недійсний або прострочений токен",
    )


@app.post("/api/auth/register")
def register(user: UserAuth):
  email = user.email.strip().lower()
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


@app.post("/api/expenses/smart-add")
def add_expense(
    payload: SmartExpensePayload, user_id: int = Depends(get_current_user_id)
):
  desc, amount = parse_raw_text(payload.raw_text)
  if amount is None or amount <= 0:
    raise HTTPException(
        status_code=400,
        detail="Вкажіть суму у тексті (наприклад: 'Сільпо 350')",
    )

  if payload.manual_category and payload.manual_category in TARGET_CATEGORIES:
    cat = payload.manual_category
  else:
    if classifier is None:
      cat = "покупки"
    else:
      cat = classifier.predict([desc])[0]

  date_now = datetime.now().strftime("%Y-%m-%d %H:%M")
  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transactions (user_id, date, description, amount,"
        " category) VALUES (?, ?, ?, ?, ?)",
        (user_id, date_now, desc, amount, cat),
    )
    conn.commit()

  return {"status": "success", "category": cat}


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
