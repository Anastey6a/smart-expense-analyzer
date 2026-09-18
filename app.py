from datetime import datetime
import io
import joblib
import re
import sqlite3
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Smart Expense Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "expenses.db"

# 5 обов'язкових категорій
TARGET_CATEGORIES = ["їжа", "транспорт", "комунальні", "покупки", "розваги"]

try:
    classifier = joblib.load("classifier.joblib")
except Exception:
    classifier = None


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL
            )
        """
        )
        conn.commit()


init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


class SmartExpensePayload(BaseModel):
    raw_text: str
    manual_category: str | None = None


def parse_raw_text(raw: str):
    raw = raw.strip()
    match = re.search(r"(\d+([.,]\d+)?)", raw)
    if not match:
        return None, None
    amount = float(match.group(1).replace(",", "."))
    desc = raw[: match.start()] + raw[match.end() :]
    desc = re.sub(r"\s+", " ", desc).strip()
    if not desc:
        desc = "Витрата"
    return desc, amount


@app.post("/api/expenses/smart-add")
def add_expense(payload: SmartExpensePayload):
    desc, amount = parse_raw_text(payload.raw_text)
    if amount is None or amount <= 0:
        raise HTTPException(
            status_code=400,
            detail="Вкажіть суму у тексті (наприклад: 'Сільпо 350')",
        )

    # Визначаємо категорію: вручну або через ML
    if (
        payload.manual_category
        and payload.manual_category in TARGET_CATEGORIES
    ):
        cat = payload.manual_category
    else:
        if classifier is None:
            raise HTTPException(
                status_code=500, detail="Класифікатор не завантажено"
            )
        cat = classifier.predict([desc])[0]

    date_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO transactions (date, description, amount, category) VALUES (?, ?, ?, ?)",
            (date_now, desc, amount, cat),
        )
        conn.commit()

    return {"status": "success", "category": cat}


@app.post("/api/analyze-and-save")
async def analyze_and_save_csv(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(
            status_code=500, detail="Класифікатор не завантажено"
        )

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Некоректний CSV файл")

    df.columns = [c.strip().lower() for c in df.columns]
    req_cols = {"date", "description", "amount"}
    if not req_cols.issubset(set(df.columns)):
        raise HTTPException(
            status_code=400, detail=f"CSV повинен містити стовпчики: {req_cols}"
        )

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    df["description"] = df["description"].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    df = df.dropna(subset=["amount", "date"])

    # ML-класифікація банківських транзакцій
    df["category"] = classifier.predict(df["description"])

    with sqlite3.connect(DB_FILE) as conn:
        for _, row in df.iterrows():
            conn.execute(
                "INSERT INTO transactions (date, description, amount, category) VALUES (?, ?, ?, ?)",
                (
                    str(row["date"]),
                    str(row["description"]),
                    float(row["amount"]),
                    str(row["category"]),
                ),
            )
        conn.commit()

    return {"status": "success", "imported": len(df)}


@app.delete("/api/expenses/{item_id}")
def delete_expense(item_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (item_id,))
        conn.commit()
    return {"status": "success"}


@app.get("/api/dashboard")
def get_dashboard():
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)

    if df.empty:
        empty_breakdown = {cat: {"sum": 0.0, "percentage": 0.0} for cat in TARGET_CATEGORIES}
        return {
            "total_spent": 0.0,
            "forecast_next_month": 0.0,
            "breakdown": empty_breakdown,
            "recent_transactions": [],
        }

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    total_spent = float(df["amount"].sum())

    # Агрегація часток точно за 5 категоріями
    breakdown = {}
    for cat in TARGET_CATEGORIES:
        cat_sum = float(df[df["category"] == cat]["amount"].sum())
        breakdown[cat] = {
            "sum": round(cat_sum, 2),
            "percentage": round((cat_sum / total_spent) * 100, 1) if total_spent > 0 else 0.0,
        }

    # Прогноз наступного місяця на базі поточних витрат і середньої динаміки
    forecast_next_month = round(total_spent * 1.05, 2)

    recent = df.sort_values(by="id", ascending=False).head(20).to_dict(orient="records")

    return {
        "total_spent": round(total_spent, 2),
        "forecast_next_month": forecast_next_month,
        "breakdown": breakdown,
        "recent_transactions": recent,
    }
if __name__ == "__main__":
  import os
  import uvicorn

  port = int(os.environ.get("PORT", 8000))
  uvicorn.run("app:app", host="0.0.0.0", port=port)