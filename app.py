import csv
import hashlib
import io
import secrets
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression


FEATURE_COLUMNS = [
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership",
]
TARGET_COLUMN = "Yearly Amount Spent"
SESSION_COOKIE_KEY = "ecommerce_session"
SESSION_STORE: dict[str, str] = {}
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "ecommerce.db"


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash.startswith("pbkdf2_sha256$"):
        return stored_hash == password
    _, salt, digest_hex = stored_hash.split("$")
    computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 200000)
    return computed.hex() == digest_hex


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                avg_session_length REAL NOT NULL,
                time_on_app REAL NOT NULL,
                time_on_website REAL NOT NULL,
                length_of_membership REAL NOT NULL,
                predicted_amount REAL NOT NULL,
                best_channel TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )

        admin_exists = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if not admin_exists:
            conn.execute(
                "INSERT INTO users (username, email, full_name, password_hash, role) VALUES (?, ?, ?, ?, ?)",
                ("admin", "admin@ecommerce.local", "System Admin", hash_password("admin123"), "admin"),
            )


init_db()


def find_dataset_path() -> Path:
    candidates = [
        BASE_DIR / "data" / "Ecommerce Customers",
        Path("C:/Users/acer/Downloads/Ecommerce Customers"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the ecommerce customer dataset. "
        "Place 'Ecommerce Customers' in the project 'data' folder or the Downloads folder."
    )


def train_model() -> tuple[LinearRegression, pd.DataFrame]:
    dataset_path = find_dataset_path()
    df = pd.read_csv(dataset_path)
    model = LinearRegression()
    model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    return model, df


MODEL, DATAFRAME = train_model()
APP_COEFFICIENT = float(MODEL.coef_[FEATURE_COLUMNS.index("Time on App")])
WEBSITE_COEFFICIENT = float(MODEL.coef_[FEATURE_COLUMNS.index("Time on Website")])
BEST_CHANNEL = "App" if abs(APP_COEFFICIENT) > abs(WEBSITE_COEFFICIENT) else "Website"

app = FastAPI(
    title="Ecommerce Customer Spend Portal",
    description="Multi-page ecommerce prediction application with authentication and admin features.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class CustomerInput(BaseModel):
    avg_session_length: float = Field(..., ge=0, description="Average session length in minutes.")
    time_on_app: float = Field(..., ge=0, description="Average time spent on the app in minutes.")
    time_on_website: float = Field(..., ge=0, description="Average time spent on the website in minutes.")
    length_of_membership: float = Field(..., ge=0, description="Length of customer membership in years.")


def get_session_username(request: Request) -> str | None:
    session_id = request.cookies.get(SESSION_COOKIE_KEY)
    if not session_id:
        return None
    return SESSION_STORE.get(session_id)


def get_current_user(request: Request) -> dict[str, Any] | None:
    username = get_session_username(request)
    if not username:
        return None
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, username, email, full_name, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def require_login(request: Request) -> dict[str, Any]:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def is_admin(user: dict[str, Any] | None) -> bool:
    return bool(user and user.get("role") == "admin")


def predict_for_features(features: list[float]) -> dict[str, Any]:
    prediction = float(MODEL.predict([features])[0])
    return {
        "predicted_yearly_amount_spent": round(prediction, 2),
        "best_channel": BEST_CHANNEL,
        "channel_coefficients": {
            "app": round(APP_COEFFICIENT, 4),
            "website": round(WEBSITE_COEFFICIENT, 4),
        },
    }


def get_recent_predictions(user_id: int, limit: int = 5) -> list[dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT avg_session_length, time_on_app, time_on_website, length_of_membership,
                   predicted_amount, best_channel, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_dashboard_context(user: dict[str, Any]) -> dict[str, Any]:
    coefficients = dict(zip(FEATURE_COLUMNS, map(float, MODEL.coef_)))
    max_abs_coefficient = max(abs(value) for value in coefficients.values()) if coefficients else 1.0
    return {
        "username": user["username"],
        "full_name": user["full_name"],
        "role": user.get("role"),
        "is_admin": is_admin(user),
        "rows": int(len(DATAFRAME)),
        "best_channel": BEST_CHANNEL,
        "summary": (
            f"The trained linear regression model suggests the {BEST_CHANNEL.lower()} "
            "has the strongest impact on yearly spend based on its coefficient magnitude."
        ),
        "coefficients": coefficients,
        "max_abs_coefficient": max_abs_coefficient,
        "intercept": float(MODEL.intercept_),
        "recent_predictions": get_recent_predictions(user["id"], limit=5),
    }


@app.get("/")
async def root(request: Request) -> RedirectResponse:
    if get_session_username(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@app.get("/login")
async def login_page(request: Request) -> Any:
    if get_session_username(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": ""})


@app.post("/login")
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> Any:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row and verify_password(password, row["password_hash"]):
        session_id = secrets.token_urlsafe(32)
        SESSION_STORE[session_id] = row["username"]
        response = RedirectResponse(url="/dashboard", status_code=303)
        response.set_cookie(key=SESSION_COOKIE_KEY, value=session_id, httponly=True, samesite="lax")
        return response

    return templates.TemplateResponse(
        request,
        "login.html",
        {"error": "Invalid username or password. Try admin / admin123"},
        status_code=401,
    )


@app.get("/register")
async def register_page(request: Request) -> Any:
    if get_session_username(request):
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse(request, "register.html", {"error": ""})


@app.post("/register")
async def register_post(
    request: Request,
    full_name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
) -> Any:
    if not username or not password:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Username and password are required."},
            status_code=400,
        )
    if password != confirm_password:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": "Passwords do not match."},
            status_code=400,
        )

    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ? OR email = ?",
            (username, email),
        ).fetchone()
        if existing:
            return templates.TemplateResponse(
                request,
                "register.html",
                {"error": "Username or email already exists."},
                status_code=400,
            )

        conn.execute(
            "INSERT INTO users (username, email, full_name, password_hash, role) VALUES (?, ?, ?, ?, 'user')",
            (username, email, full_name.strip(), hash_password(password)),
        )

    session_id = secrets.token_urlsafe(32)
    SESSION_STORE[session_id] = username
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key=SESSION_COOKIE_KEY, value=session_id, httponly=True, samesite="lax")
    return response


@app.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    session_id = request.cookies.get(SESSION_COOKIE_KEY)
    if session_id:
        SESSION_STORE.pop(session_id, None)
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key=SESSION_COOKIE_KEY)
    return response


@app.get("/dashboard")
async def dashboard(request: Request) -> Any:
    user = require_login(request)
    return templates.TemplateResponse(request, "dashboard.html", get_dashboard_context(user))


@app.get("/profile")
async def profile_page(request: Request) -> Any:
    user = require_login(request)
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT id, username, email, full_name, role FROM users WHERE id = ?",
            (user["id"],),
        ).fetchone()
    return templates.TemplateResponse(
        request,
        "profile.html",
        {
            "user": dict(row),
            "success": "",
            "error": "",
        },
    )


@app.post("/profile")
async def profile_update(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str | None = Form(default=None),
    confirm_password: str | None = Form(default=None),
) -> Any:
    user = require_login(request)
    if password and password != confirm_password:
        return templates.TemplateResponse(
            request,
            "profile.html",
            {"user": user, "success": "", "error": "Passwords do not match."},
            status_code=400,
        )

    with get_db_connection() as conn:
        if password:
            conn.execute(
                "UPDATE users SET full_name = ?, email = ?, password_hash = ? WHERE id = ?",
                (full_name.strip(), email.strip(), hash_password(password), user["id"]),
            )
        else:
            conn.execute(
                "UPDATE users SET full_name = ?, email = ? WHERE id = ?",
                (full_name.strip(), email.strip(), user["id"]),
            )

    user = dict(get_current_user(request) or {})
    user["full_name"] = full_name.strip()
    user["email"] = email.strip()
    return templates.TemplateResponse(
        request,
        "profile.html",
        {"user": user, "success": "Profile updated successfully.", "error": ""},
    )


@app.post("/admin/user/{user_id}/role")
async def update_user_role(
    request: Request,
    user_id: int,
    role: str = Form(...),
) -> RedirectResponse:
    current_user = require_login(request)
    if not is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    if role not in {"user", "admin"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role selected")

    with get_db_connection() as conn:
        target = conn.execute("SELECT id, role FROM users WHERE id = ?", (user_id,)).fetchone()
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        if target["role"] == "admin" and role == "user":
            admin_count = conn.execute("SELECT COUNT(*) AS total FROM users WHERE role = 'admin'").fetchone()["total"]
            if admin_count <= 1:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one admin account must remain")
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))

    return RedirectResponse(url="/admin", status_code=303)


@app.post("/admin/user/{user_id}/delete")
async def delete_user(request: Request, user_id: int) -> RedirectResponse:
    current_user = require_login(request)
    if not is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    if user_id == current_user["id"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot delete your own admin account")

    with get_db_connection() as conn:
        target = conn.execute("SELECT id, role FROM users WHERE id = ?", (user_id,)).fetchone()
        if not target:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        if target["role"] == "admin":
            admin_count = conn.execute("SELECT COUNT(*) AS total FROM users WHERE role = 'admin'").fetchone()["total"]
            if admin_count <= 1:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one admin account must remain")
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    return RedirectResponse(url="/admin", status_code=303)


@app.get("/admin")
async def admin_page(request: Request) -> Any:
    user = require_login(request)
    if not is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    with get_db_connection() as conn:
        users = conn.execute(
            "SELECT id, username, email, full_name, role, created_at FROM users ORDER BY id"
        ).fetchall()
        stats = conn.execute(
            """
            SELECT COUNT(*) AS total_users,
                   SUM(CASE WHEN role='admin' THEN 1 ELSE 0 END) AS total_admins,
                   SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) AS total_customers,
                   (SELECT COUNT(*) FROM predictions) AS total_predictions
            FROM users
            """
        ).fetchone()
    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "user": user,
            "users": [dict(row) for row in users],
            "stats": dict(stats),
        },
    )


@app.get("/predict")
async def predict_page(request: Request) -> Any:
    user = require_login(request)
    return templates.TemplateResponse(
        request,
        "predict.html",
        {"user": user, "result": None, "error": ""},
    )


@app.post("/predict")
async def predict_submit(
    request: Request,
    avg_session_length: float = Form(...),
    time_on_app: float = Form(...),
    time_on_website: float = Form(...),
    length_of_membership: float = Form(...),
) -> Any:
    user = require_login(request)
    features = [avg_session_length, time_on_app, time_on_website, length_of_membership]
    result = predict_for_features(features)

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO predictions (user_id, avg_session_length, time_on_app, time_on_website, length_of_membership, predicted_amount, best_channel) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user["id"], avg_session_length, time_on_app, time_on_website, length_of_membership, result["predicted_yearly_amount_spent"], result["best_channel"]),
        )

    return templates.TemplateResponse(
        request,
        "predict.html",
        {"user": user, "result": result, "error": ""},
    )


@app.get("/upload")
async def upload_page(request: Request) -> Any:
    user = require_login(request)
    return templates.TemplateResponse(request, "upload.html", {"user": user, "result": None, "error": ""})


@app.post("/upload")
async def upload_csv(request: Request, file: UploadFile = File(...)) -> Any:
    user = require_login(request)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return templates.TemplateResponse(
            request,
            "upload.html",
            {"user": user, "result": None, "error": "Please upload a valid CSV file."},
            status_code=400,
        )

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {"user": user, "result": None, "error": "Unable to parse CSV. Check the column names and file format."},
            status_code=400,
        )

    normalized = df.copy()
    rename_map = {
        "avg.session.length": "Avg. Session Length",
        "avg_session_length": "Avg. Session Length",
        "timeonapp": "Time on App",
        "time_on_app": "Time on App",
        "timeonwebsite": "Time on Website",
        "time_on_website": "Time on Website",
        "lengthofmembership": "Length of Membership",
        "length_of_membership": "Length of Membership",
    }
    normalized.columns = [str(col).strip() for col in normalized.columns]
    normalized = normalized.rename(columns=lambda c: rename_map.get(c.lower().replace(" ", "").replace("-", ""), c))

    missing = [col for col in FEATURE_COLUMNS if col not in normalized.columns]
    if missing:
        return templates.TemplateResponse(
            request,
            "upload.html",
            {"user": user, "result": None, "error": f"Missing required columns: {', '.join(missing)}"},
            status_code=400,
        )

    results = []
    for _, row in normalized[FEATURE_COLUMNS].iterrows():
        features = [float(row[col]) for col in FEATURE_COLUMNS]
        prediction = predict_for_features(features)
        results.append(
            {
                "avg_session_length": round(float(row[FEATURE_COLUMNS[0]]), 2),
                "time_on_app": round(float(row[FEATURE_COLUMNS[1]]), 2),
                "time_on_website": round(float(row[FEATURE_COLUMNS[2]]), 2),
                "length_of_membership": round(float(row[FEATURE_COLUMNS[3]]), 2),
                "predicted_amount": prediction["predicted_yearly_amount_spent"],
                "best_channel": prediction["best_channel"],
            }
        )

    with get_db_connection() as conn:
        for item in results:
            conn.execute(
                "INSERT INTO predictions (user_id, avg_session_length, time_on_app, time_on_website, length_of_membership, predicted_amount, best_channel) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user["id"], item["avg_session_length"], item["time_on_app"], item["time_on_website"], item["length_of_membership"], item["predicted_amount"], item["best_channel"]),
            )

    return templates.TemplateResponse(
        request,
        "upload.html",
        {"user": user, "result": results, "error": ""},
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-summary")
def model_summary() -> dict[str, Any]:
    coefficients = dict(zip(FEATURE_COLUMNS, map(float, MODEL.coef_)))
    return {
        "rows": int(len(DATAFRAME)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "intercept": float(MODEL.intercept_),
        "coefficients": coefficients,
        "best_channel": BEST_CHANNEL,
        "summary": (
            f"The trained linear regression model suggests the {BEST_CHANNEL.lower()} "
            "has the strongest impact on yearly spend based on its coefficient magnitude."
        ),
    }


@app.post("/api/predict")
def predict_customer_spend(customer: CustomerInput) -> dict[str, Any]:
    features = [
        customer.avg_session_length,
        customer.time_on_app,
        customer.time_on_website,
        customer.length_of_membership,
    ]
    result = predict_for_features(features)
    return {
        "predicted_yearly_amount_spent": result["predicted_yearly_amount_spent"],
        "best_channel": result["best_channel"],
        "channel_coefficients": result["channel_coefficients"],
        "input": {
            "avg_session_length": customer.avg_session_length,
            "time_on_app": customer.time_on_app,
            "time_on_website": customer.time_on_website,
            "length_of_membership": customer.length_of_membership,
        },
    }
