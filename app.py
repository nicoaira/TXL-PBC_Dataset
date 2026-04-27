"""Streamlit app for TXL-PBC YOLO26 blood-cell detection.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INITIAL_CREDITS = 10  # free credits granted to every new user
AUTH_COOKIE_NAME = "txl_pbc_refresh_token"
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

# ---------------------------------------------------------------------------
# Firebase auth
# ---------------------------------------------------------------------------

_AUTH_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

div[data-testid="stVerticalBlock"] .auth-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.auth-logo {
    font-size: 3.5rem;
    text-align: center;
    margin-bottom: 0.2rem;
}
.auth-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}
.auth-subtitle {
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.88rem;
    margin-bottom: 1.8rem;
}
.auth-divider {
    display: flex;
    align-items: center;
    color: rgba(255,255,255,0.3);
    font-size: 0.82rem;
    margin: 1.2rem 0;
}
.auth-divider::before, .auth-divider::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    margin: 0 0.6rem;
}
/* Style the submit buttons */
div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
</style>
"""


# ---------------------------------------------------------------------------
# Credit system — Firebase Firestore REST API
# ---------------------------------------------------------------------------

def _firestore_base_url() -> str:
    project_id = st.secrets["firebase"]["project_id"]
    return (
        f"https://firestore.googleapis.com/v1/projects/{project_id}"
        "/databases/(default)/documents"
    )


def _get_admin_firestore_client():
    try:
        _ = st.secrets["firebase_admin"]
    except KeyError:
        return None

    try:
        from firebase_admin import firestore as fb_fs

        return fb_fs.client(app=_firebase_admin_app())
    except Exception:
        return None


def _get_user_credits(user_id: str, id_token: str) -> int:
    admin_db = _get_admin_firestore_client()
    if admin_db is not None:
        doc_ref = admin_db.collection("users").document(user_id)
        doc = doc_ref.get()
        if not doc.exists:
            doc_ref.set({"credits": INITIAL_CREDITS}, merge=True)
            return INITIAL_CREDITS

        credits = (doc.to_dict() or {}).get("credits")
        if credits is None:
            doc_ref.set({"credits": INITIAL_CREDITS}, merge=True)
            return INITIAL_CREDITS
        return int(credits)

    import requests
    url = f"{_firestore_base_url()}/users/{user_id}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=10,
        )
    except Exception:
        return st.session_state.get("credits") or 0

    if resp.status_code == 404:
        _save_user_credits(user_id, id_token, INITIAL_CREDITS)
        return INITIAL_CREDITS
    if resp.status_code != 200:
        return st.session_state.get("credits") or 0

    data = resp.json()
    val = data.get("fields", {}).get("credits", {}).get("integerValue")
    return int(val) if val is not None else INITIAL_CREDITS


def _save_user_credits(user_id: str, id_token: str, credits: int) -> bool:
    admin_db = _get_admin_firestore_client()
    if admin_db is not None:
        admin_db.collection("users").document(user_id).set(
            {"credits": int(credits)}, merge=True
        )
        return True

    import requests
    url = f"{_firestore_base_url()}/users/{user_id}?updateMask.fieldPaths=credits"
    payload = {"fields": {"credits": {"integerValue": str(credits)}}}
    try:
        resp = requests.patch(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _refresh_credits() -> None:
    user = st.session_state.get("user")
    if not user:
        st.session_state.credits = 0
        return
    st.session_state.credits = _get_user_credits(
        user.get("localId", ""), user.get("idToken", "")
    )


def _deduct_credits(amount: int) -> bool:
    """Deduct credits from the current user. Returns False if balance is insufficient."""
    user = st.session_state.get("user")
    if not user:
        return False
    current = st.session_state.get("credits") or 0
    if current < amount:
        return False
    new_bal = current - amount
    ok = _save_user_credits(user["localId"], user["idToken"], new_bal)
    if ok:
        st.session_state.credits = new_bal
    return ok


# ---------------------------------------------------------------------------
# Admin — helpers
# ---------------------------------------------------------------------------

def _get_admin_emails() -> list[str]:
    try:
        val = st.secrets.get("app", {}).get("admin_emails", [])
        if isinstance(val, str):
            return [val]
        return list(val)
    except Exception:
        return []


def _is_admin() -> bool:
    user = st.session_state.get("user")
    if not user:
        return False
    return user.get("email", "") in _get_admin_emails()


@st.cache_resource
def _firebase_admin_app():
    import firebase_admin
    from firebase_admin import credentials

    cfg = dict(st.secrets["firebase_admin"])
    if "private_key" in cfg:
        cfg["private_key"] = cfg["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cfg)
    try:
        return firebase_admin.get_app("txl_admin")
    except ValueError:
        return firebase_admin.initialize_app(cred, name="txl_admin")


def _admin_list_users() -> list[dict[str, Any]]:
    from firebase_admin import auth as fb_auth
    from firebase_admin import firestore as fb_fs

    app = _firebase_admin_app()
    db = fb_fs.client(app=app)

    users: list[dict[str, Any]] = []
    page = fb_auth.list_users(app=app)
    while page:
        for u in page.users:
            doc = db.collection("users").document(u.uid).get()
            doc_data = doc.to_dict() if doc.exists else {}
            users.append({
                "uid": u.uid,
                "email": u.email or "",
                "credits": doc_data.get("credits", 0),
            })
        page = page.get_next_page()
    return users


def _admin_set_user_credits(uid: str, credits: int) -> None:
    from firebase_admin import firestore as fb_fs

    app = _firebase_admin_app()
    db = fb_fs.client(app=app)
    db.collection("users").document(uid).set({"credits": credits}, merge=True)


# ---------------------------------------------------------------------------
# Stripe Payments
# ---------------------------------------------------------------------------

def _create_stripe_checkout(credits_to_buy: int, price_cents: int) -> None:
    import stripe
    try:
        stripe.api_key = st.secrets["stripe"]["api_key"]
    except KeyError:
        st.error("Stripe is not configured.")
        return

    user = st.session_state.get("user")
    if not user:
        return

    success_url = "http://localhost:8501/?session_id={CHECKOUT_SESSION_ID}"
    cancel_url = "http://localhost:8501/"

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": f"{credits_to_buy} Credits",
                    },
                    "unit_amount": price_cents,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "user_id": user["localId"],
                "credits": str(credits_to_buy),
            }
        )
        st.markdown(f'<meta http-equiv="refresh" content="0;url={session.url}">', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to create checkout session: {e}")


def _check_stripe_payment() -> None:
    session_id = st.query_params.get("session_id")
    if not session_id:
        return

    import stripe
    try:
        stripe.api_key = st.secrets["stripe"]["api_key"]
    except KeyError:
        return

    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == "paid":
            if session.metadata.get("processed") != "true":
                user_id = session.metadata.get("user_id")
                credits_to_add = int(session.metadata.get("credits", 0))

                user = st.session_state.get("user")
                if user and user.get("localId") == user_id:
                    current = _get_user_credits(user["localId"], user["idToken"])
                    new_bal = current + credits_to_add
                    if _save_user_credits(user["localId"], user["idToken"], new_bal):
                        st.session_state.credits = new_bal
                        stripe.checkout.Session.modify(
                            session_id,
                            metadata={**session.metadata, "processed": "true"}
                        )
                        st.success(f"Payment successful! Added {credits_to_add} credits.")
                        st.query_params.clear()
                    else:
                        st.error("Failed to add credits to your account. Please contact support.")
    except Exception as e:
        st.error(f"Error verifying payment: {e}")


# ---------------------------------------------------------------------------
# Firebase auth — login / register / Google OAuth
# ---------------------------------------------------------------------------

@st.cache_resource
def _firebase_auth():
    import pyrebase
    cfg = st.secrets["firebase"]
    config = {
        "apiKey": cfg["api_key"],
        "authDomain": cfg["auth_domain"],
        "projectId": cfg["project_id"],
        "storageBucket": cfg["storage_bucket"],
        "messagingSenderId": cfg["messaging_sender_id"],
        "appId": cfg["app_id"],
        "databaseURL": "",
    }
    return pyrebase.initialize_app(config).auth()


def _cookie_controller():
    try:
        from streamlit_cookies_controller import CookieController
    except ImportError:
        return None
    return CookieController()


def _set_auth_cookie(refresh_token: str | None) -> None:
    if not refresh_token:
        return

    controller = _cookie_controller()
    if controller is None:
        return

    from datetime import datetime, timedelta

    controller.set(
        AUTH_COOKIE_NAME,
        refresh_token,
        expires=datetime.now() + timedelta(seconds=AUTH_COOKIE_MAX_AGE),
        max_age=AUTH_COOKIE_MAX_AGE,
        secure=False,
        same_site="strict",
    )


def _clear_auth_cookie() -> None:
    controller = _cookie_controller()
    if controller is None:
        return
    controller.remove(AUTH_COOKIE_NAME)


def _lookup_firebase_user(id_token: str) -> dict[str, Any]:
    import requests

    api_key = st.secrets["firebase"]["api_key"]
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={api_key}"
    resp = requests.post(url, json={"idToken": id_token}, timeout=10)
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Firebase user lookup failed."))

    users = data.get("users") or []
    if not users:
        raise RuntimeError("Firebase user lookup returned no users.")
    return users[0]


def _restore_login_from_cookie() -> bool:
    if st.session_state.get("user"):
        return True

    controller = _cookie_controller()
    if controller is None:
        return False

    refresh_token = controller.get(AUTH_COOKIE_NAME)
    if not refresh_token:
        return False

    import requests

    api_key = st.secrets["firebase"]["api_key"]
    url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"
    try:
        resp = requests.post(
            url,
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
            timeout=10,
        )
        data = resp.json()
        if "error" in data:
            _clear_auth_cookie()
            return False

        id_token = data["id_token"]
        user_info = _lookup_firebase_user(id_token)
        st.session_state.user = {
            "email": user_info.get("email", ""),
            "localId": user_info.get("localId") or data.get("user_id"),
            "idToken": id_token,
            "refreshToken": data.get("refresh_token", refresh_token),
            "displayName": user_info.get("displayName", ""),
        }
        st.session_state.auth_error = None
        _set_auth_cookie(st.session_state.user.get("refreshToken"))
        _refresh_credits()
        return True
    except Exception:
        _clear_auth_cookie()
        return False


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "user": None,
        "auth_error": None,
        "credits": None,
        "show_admin": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _do_login(email: str, password: str) -> None:
    try:
        user = _firebase_auth().sign_in_with_email_and_password(email, password)
        st.session_state.user = user
        st.session_state.auth_error = None
        _set_auth_cookie(user.get("refreshToken"))
        _refresh_credits()
    except Exception as exc:
        msg = str(exc)
        if any(k in msg for k in ("INVALID_PASSWORD", "EMAIL_NOT_FOUND", "INVALID_LOGIN_CREDENTIALS")):
            st.session_state.auth_error = "Invalid email or password."
        else:
            st.session_state.auth_error = "Login failed. Please try again."


def _do_register(email: str, password: str) -> None:
    try:
        user = _firebase_auth().create_user_with_email_and_password(email, password)
        st.session_state.user = user
        st.session_state.auth_error = None
        _set_auth_cookie(user.get("refreshToken"))
        _refresh_credits()
    except Exception as exc:
        msg = str(exc)
        if "EMAIL_EXISTS" in msg:
            st.session_state.auth_error = "An account with this email already exists."
        elif "WEAK_PASSWORD" in msg:
            st.session_state.auth_error = "Password must be at least 6 characters."
        else:
            st.session_state.auth_error = "Registration failed. Please try again."


def _do_google_signin() -> None:
    import requests
    try:
        from streamlit_oauth import OAuth2Component
    except ImportError:
        st.session_state.auth_error = "streamlit-oauth not installed."
        return

    try:
        g = st.secrets["google"]
        redirect_uri = st.secrets.get("app", {}).get("redirect_uri", "http://localhost:8501")
    except KeyError:
        st.warning("Google sign-in is not configured yet.")
        return

    oauth2 = OAuth2Component(
        client_id=g["client_id"],
        client_secret=g["client_secret"],
        authorize_endpoint="https://accounts.google.com/o/oauth2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        refresh_token_endpoint="https://oauth2.googleapis.com/token",
        revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
    )
    result = oauth2.authorize_button(
        name="Continue with Google",
        redirect_uri=redirect_uri,
        scope="openid email profile",
        key="google_oauth",
        icon="https://www.google.com/favicon.ico",
        use_container_width=True,
        extras_params={"prompt": "select_account"},
    )
    if result and "token" in result:
        id_token = result["token"].get("id_token", "")
        api_key = st.secrets["firebase"]["api_key"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key={api_key}"
        payload = {
            "postBody": f"id_token={id_token}&providerId=google.com",
            "requestUri": redirect_uri,
            "returnSecureToken": True,
            "returnIdpCredential": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            data = resp.json()
            if "error" in data:
                st.session_state.auth_error = data["error"].get("message", "Google sign-in failed.")
            else:
                st.session_state.user = {
                    "email": data.get("email"),
                    "localId": data.get("localId"),
                    "idToken": data.get("idToken"),
                    "refreshToken": data.get("refreshToken"),
                    "displayName": data.get("displayName", ""),
                }
                st.session_state.auth_error = None
                _set_auth_cookie(data.get("refreshToken"))
                _refresh_credits()
                st.rerun()
        except Exception as exc:
            st.session_state.auth_error = f"Google sign-in error: {exc}"


def _do_logout() -> None:
    st.session_state.user = None
    st.session_state.auth_error = None
    st.session_state.credits = None
    _clear_auth_cookie()


def render_auth_page() -> None:
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="auth-logo">🔬</div>', unsafe_allow_html=True)
        st.markdown('<p class="auth-title">TXL-PBC Detector</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="auth-subtitle">Blood cell detection powered by YOLO26</p>',
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs(["Sign in", "Create account"])

        with tab_login:
            with st.form("login_form", border=False):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
            if submitted:
                _do_login(email, password)
                st.rerun()

        with tab_register:
            with st.form("register_form", border=False):
                email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                password = st.text_input("Password", type="password", placeholder="Min. 6 characters", key="reg_pw")
                password2 = st.text_input("Confirm password", type="password", placeholder="Repeat password", key="reg_pw2")
                submitted = st.form_submit_button("Create account", use_container_width=True, type="primary")
            if submitted:
                if password != password2:
                    st.session_state.auth_error = "Passwords do not match."
                else:
                    _do_register(email, password)
                    st.rerun()

        if st.session_state.auth_error:
            st.error(st.session_state.auth_error)

        st.markdown('<div class="auth-divider">or</div>', unsafe_allow_html=True)
        _do_google_signin()
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/best_granular_wbc_v3.pt"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models/best_granular_wbc.pt"
CLASS_NAMES = [
    "RBC", "Platelets",
    "Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil",
]
CLASS_COLORS = {
    "WBC": (22, 163, 74),
    "RBC": (220, 38, 38),
    "Platelets": (37, 99, 235),
    "Neutrophil": (16, 185, 129),
    "Lymphocyte": (234, 88, 12),
    "Monocyte": (168, 85, 247),
    "Eosinophil": (236, 72, 153),
    "Basophil": (250, 204, 21),
}


@st.cache_resource(show_spinner="Loading YOLO26 checkpoint...")
def load_model(model_path: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Failed to import ultralytics: {exc!r}") from exc
    return YOLO(model_path)


def resolve_model_path(user_override: str | None) -> Path:
    if user_override:
        candidate = Path(user_override).expanduser()
        if candidate.exists():
            return candidate.resolve()
        st.warning(f"Override model not found: {candidate}. Falling back to defaults.")
    for candidate in (DEFAULT_MODEL_PATH, FALLBACK_MODEL_PATH):
        if candidate.exists():
            return candidate.resolve()
    for found in sorted(
        (PROJECT_ROOT / "runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        return found.resolve()
    raise FileNotFoundError(
        f"No checkpoint found at {DEFAULT_MODEL_PATH} or under runs/**/weights/best.pt"
    )


def class_names_from_model(model: Any) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict) and names:
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    return CLASS_NAMES


def _font(size: int = 16) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    boxes_xyxy: list[tuple[float, float, float, float]],
    confs: list[float],
    classes: list[int],
    names: list[str],
    line_width: int = 3,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _font(16)
    for box, conf, class_id in zip(boxes_xyxy, confs, classes, strict=False):
        name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        color = CLASS_COLORS.get(name, (147, 51, 234))
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{name} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        text_y = max(0, y1 - text_h - 2 * pad)
        draw.rectangle(
            (x1, text_y, x1 + text_w + 2 * pad, text_y + text_h + 2 * pad),
            fill=color,
        )
        draw.text((x1 + pad, text_y + pad), label, fill=(255, 255, 255), font=font)
    return annotated


def run_inference(
    model: Any,
    image: Image.Image,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
) -> dict[str, Any]:
    predict_kwargs: dict[str, Any] = {
        "source": np.asarray(image.convert("RGB")),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "verbose": False,
        "save": False,
    }
    if device:
        predict_kwargs["device"] = device
    results = model.predict(**predict_kwargs)
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return {"boxes": [], "confs": [], "classes": []}
    return {
        "boxes": [tuple(float(v) for v in box) for box in boxes.xyxy.cpu().tolist()],
        "confs": [float(c) for c in boxes.conf.cpu().tolist()],
        "classes": [int(c) for c in boxes.cls.cpu().tolist()],
    }


def render_detection_summary(
    detections: dict[str, Any],
    names: list[str],
) -> None:
    total = len(detections["boxes"])
    counts: dict[str, int] = {name: 0 for name in names}
    for class_id in detections["classes"]:
        if 0 <= class_id < len(names):
            counts[names[class_id]] += 1

    st.metric("Total detections", total)
    if total:
        cols = st.columns(len(names))
        for col, name in zip(cols, names, strict=False):
            col.metric(name, counts.get(name, 0))

        with st.expander("Detection details", expanded=False):
            rows = [
                {
                    "class": names[c] if 0 <= c < len(names) else str(c),
                    "confidence": round(conf, 3),
                    "x1": round(box[0], 1),
                    "y1": round(box[1], 1),
                    "x2": round(box[2], 1),
                    "y2": round(box[3], 1),
                }
                for box, conf, c in zip(
                    detections["boxes"],
                    detections["confs"],
                    detections["classes"],
                    strict=False,
                )
            ]
            st.dataframe(rows)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _yolo_label_txt(
    detections: dict[str, Any],
    img_w: int,
    img_h: int,
) -> str:
    lines = []
    for box, cls_id in zip(detections["boxes"], detections["classes"], strict=False):
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def make_training_export_zip(
    entries: list[tuple[str, Image.Image, dict[str, Any]]],
    names: list[str],
) -> bytes:
    """Build a CVAT YOLO 1.1 compatible training ZIP.

    Structure:
        obj.names          - class names
        obj.data           - darknet config (required by CVAT)
        train.txt          - list of image paths
        obj_train_data/
            image1.png
            image1.txt     - YOLO label
            ...
    """
    buf = io.BytesIO()
    obj_data = (
        f"classes = {len(names)}\n"
        f"names = obj.names\n"
        f"train = train.txt\n"
        f"backup = backup/\n"
    )
    image_paths = [f"obj_train_data/{Path(n).stem}.png" for n, _, _ in entries]

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("obj.names", "\n".join(names))
        zf.writestr("obj.data", obj_data)
        zf.writestr("train.txt", "\n".join(image_paths))
        for orig_name, image, detections in entries:
            stem = Path(orig_name).stem
            img_w, img_h = image.size
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            zf.writestr(f"obj_train_data/{stem}.png", img_bytes.getvalue())
            zf.writestr(f"obj_train_data/{stem}.txt", _yolo_label_txt(detections, img_w, img_h))
    return buf.getvalue()


def sidebar_controls() -> dict[str, Any]:
    st.sidebar.header("Model")
    default_path = str(DEFAULT_MODEL_PATH)
    model_path_input = st.sidebar.text_input(
        "Checkpoint path",
        value=default_path,
        help="Path to a YOLO .pt checkpoint (relative to the project root).",
    )

    st.sidebar.header("Inference")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.1, 0.95, 0.7, 0.05)
    imgsz = st.sidebar.select_slider(
        "Image size", options=[320, 416, 512, 640, 768, 960, 1280], value=640
    )
    device = st.sidebar.selectbox(
        "Device",
        options=["auto", "cpu", "0", "0,1"],
        index=0,
        help="`auto` lets Ultralytics pick. Use `cpu` if no GPU is available.",
    )
    return {
        "model_path": model_path_input.strip() or default_path,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "device": None if device == "auto" else device,
    }


# ---------------------------------------------------------------------------
# Detection modes
# ---------------------------------------------------------------------------

def _credits_warning(needed: int) -> bool:
    """Show error and return True if the user cannot afford `needed` credits."""
    available = st.session_state.get("credits") or 0
    if available < needed:
        st.error(
            f"Not enough credits. You need **{needed}** but have **{available}**. "
            "Contact support to get more credits."
        )
        return True
    return False


def mode_camera(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Camera snapshot")
    st.caption("Cost: **1 credit** per snapshot.")

    snapshot = st.camera_input("Take a photo")
    if snapshot is None:
        return

    if _credits_warning(1):
        return

    if not _deduct_credits(1):
        st.error("Could not deduct credits. Please refresh and try again.")
        return

    image = Image.open(snapshot).convert("RGB")
    detections = run_inference(
        model, image,
        imgsz=settings["imgsz"], conf=settings["conf"],
        iou=settings["iou"], device=settings["device"],
    )
    annotated = draw_detections(image, detections["boxes"], detections["confs"], detections["classes"], names)
    left, right = st.columns(2)
    left.image(image, caption="Snapshot")
    right.image(annotated, caption="Predictions")
    render_detection_summary(detections, names)
    st.download_button(
        "Download annotated image",
        data=image_to_png_bytes(annotated),
        file_name="prediction.png",
        mime="image/png",
    )


def mode_upload_image(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Upload image")
    st.caption("Cost: **1 credit** per image.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image")

    if _credits_warning(1):
        return

    if not st.button("Run inference (1 credit)", type="primary", key="run_image"):
        return

    if not _deduct_credits(1):
        st.error("Could not deduct credits. Please refresh and try again.")
        return

    detections = run_inference(
        model, image,
        imgsz=settings["imgsz"], conf=settings["conf"],
        iou=settings["iou"], device=settings["device"],
    )
    annotated = draw_detections(image, detections["boxes"], detections["confs"], detections["classes"], names)
    left, right = st.columns(2)
    left.image(image, caption="Original")
    right.image(annotated, caption="Predictions")
    render_detection_summary(detections, names)
    col1, col2 = st.columns(2)
    col1.download_button(
        "Download annotated image",
        data=image_to_png_bytes(annotated),
        file_name=f"{Path(uploaded.name).stem}_pred.png",
        mime="image/png",
    )
    col2.download_button(
        "Export for training (YOLO)",
        data=make_training_export_zip([(uploaded.name, image, detections)], names),
        file_name=f"{Path(uploaded.name).stem}_yolo_export.zip",
        mime="application/zip",
    )


def mode_batch_images(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Batch images")
    st.caption("Cost: **1 credit** per image.")

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        return

    n = len(uploaded_files)
    st.info(f"{n} image{'s' if n != 1 else ''} selected — costs **{n} credit{'s' if n != 1 else ''}**.")

    if _credits_warning(n):
        return

    if not st.button(f"Run inference on {n} image{'s' if n != 1 else ''} ({n} credits)", type="primary", key="run_batch"):
        return

    if not _deduct_credits(n):
        st.error("Could not deduct credits. Please refresh and try again.")
        return

    zip_buffer = io.BytesIO()
    training_buffer = io.BytesIO()
    total_counts: dict[str, int] = {name: 0 for name in names}
    total_detections = 0

    progress = st.progress(0.0, text="Processing…")
    with (
        zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf_annotated,
        zipfile.ZipFile(training_buffer, "w", zipfile.ZIP_DEFLATED) as zf_training,
    ):
        obj_data = (
            f"classes = {len(names)}\n"
            f"names = obj.names\n"
            f"train = train.txt\n"
            f"backup = backup/\n"
        )
        zf_training.writestr("obj.names", "\n".join(names))
        zf_training.writestr("obj.data", obj_data)
        image_stems: list[str] = []

        for i, uploaded in enumerate(uploaded_files):
            image = Image.open(uploaded).convert("RGB")
            detections = run_inference(
                model, image,
                imgsz=settings["imgsz"], conf=settings["conf"],
                iou=settings["iou"], device=settings["device"],
            )
            annotated = draw_detections(
                image, detections["boxes"], detections["confs"], detections["classes"], names
            )
            for cls_id in detections["classes"]:
                if 0 <= cls_id < len(names):
                    total_counts[names[cls_id]] += 1
            total_detections += len(detections["boxes"])

            stem = Path(uploaded.name).stem
            img_w, img_h = image.size
            image_stems.append(stem)

            zf_annotated.writestr(f"{stem}_pred.png", image_to_png_bytes(annotated))

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            zf_training.writestr(f"obj_train_data/{stem}.png", img_bytes.getvalue())
            zf_training.writestr(f"obj_train_data/{stem}.txt", _yolo_label_txt(detections, img_w, img_h))

            with st.expander(f"{uploaded.name} — {len(detections['boxes'])} detection(s)", expanded=False):
                left, right = st.columns(2)
                left.image(image, caption="Original")
                right.image(annotated, caption="Predictions")

            progress.progress((i + 1) / n, text=f"{i + 1}/{n} processed…")

        zf_training.writestr(
            "train.txt",
            "\n".join(f"obj_train_data/{s}.png" for s in image_stems),
        )

    progress.empty()

    st.success(f"Done. {total_detections} total detections across {n} images.")
    cols = st.columns(len(names))
    for col, name in zip(cols, names, strict=False):
        col.metric(name, total_counts.get(name, 0))

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download all annotated images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="batch_predictions.zip",
        mime="application/zip",
    )
    col2.download_button(
        "Export for training (YOLO)",
        data=training_buffer.getvalue(),
        file_name="batch_yolo_export.zip",
        mime="application/zip",
    )


def mode_upload_video(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Upload video")
    st.caption("Cost: **1 credit per second** of video duration.")

    uploaded = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        return

    try:
        import av
    except ImportError:
        st.error("Video mode requires `av`. It should be installed via requirements.txt.")
        return

    video_bytes = uploaded.read()

    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            v_stream = container.streams.video[0]
            if v_stream.duration and v_stream.time_base:
                duration_secs = float(v_stream.duration * v_stream.time_base)
            elif container.duration:
                duration_secs = float(container.duration) / 1_000_000
            else:
                duration_secs = None
            fps = float(v_stream.average_rate) if v_stream.average_rate else 25.0
    except Exception as exc:
        st.error(f"Could not read video metadata: {exc}")
        return

    if duration_secs is None or duration_secs <= 0:
        st.warning("Could not determine video duration.")
        return

    credits_needed = max(1, int(duration_secs))
    available = st.session_state.get("credits") or 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{duration_secs:.1f}s")
    col2.metric("Cost", f"{credits_needed} credits")
    col3.metric("Your balance", f"{available} credits")

    if _credits_warning(credits_needed):
        return

    if not st.button(f"Run inference on video ({credits_needed} credits)", type="primary", key="run_video"):
        return

    if not _deduct_credits(credits_needed):
        st.error("Could not deduct credits. Please refresh and try again.")
        return

    # Sample at 2 fps to keep processing time reasonable
    sample_fps = min(2.0, fps)
    frame_interval = max(1, int(fps / sample_fps))

    progress = st.progress(0.0, text="Processing video…")
    annotated_frames: list[Image.Image] = []
    total_counts: dict[str, int] = {name: 0 for name in names}
    processed = 0

    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            all_frames = list(container.decode(video=0))
            total_frames = len(all_frames)
            sample_indices = set(range(0, total_frames, frame_interval))

            for i, frame in enumerate(all_frames):
                if i not in sample_indices:
                    continue
                pil_frame = frame.to_image().convert("RGB")
                detections = run_inference(
                    model, pil_frame,
                    imgsz=settings["imgsz"], conf=settings["conf"],
                    iou=settings["iou"], device=settings["device"],
                )
                annotated = draw_detections(
                    pil_frame,
                    detections["boxes"], detections["confs"],
                    detections["classes"], names,
                )
                annotated_frames.append(annotated)
                for cls_id in detections["classes"]:
                    if 0 <= cls_id < len(names):
                        total_counts[names[cls_id]] += 1
                processed += 1
                progress.progress(
                    min(1.0, (i + 1) / max(1, total_frames)),
                    text=f"Frame {i + 1}/{total_frames}…",
                )
    except Exception as exc:
        st.error(f"Error during video processing: {exc}")
        return

    progress.empty()
    st.success(f"Done. Analysed {processed} sampled frames from {duration_secs:.1f}s of video.")

    total = sum(total_counts.values())
    st.metric("Total detections (all sampled frames)", total)
    if total:
        cols = st.columns(len(names))
        for col, name in zip(cols, names, strict=False):
            col.metric(name, total_counts[name])

    if annotated_frames:
        st.subheader("Sample annotated frames")
        show = annotated_frames[::max(1, len(annotated_frames) // 8)][:8]
        cols = st.columns(min(4, len(show)))
        for col, frame in zip(cols * 2, show, strict=False):
            col.image(frame)


def mode_live(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Live webcam (WebRTC)")
    st.caption("Cost: **1 credit per second** of streaming. Credits are deducted as you stream.")

    if (st.session_state.get("credits") or 0) < 1:
        st.error("You have no credits left. You need at least 1 credit to start streaming.")
        return

    try:
        import av  # noqa: F401
        from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
    except ImportError:
        st.error(
            "Live mode needs `streamlit-webrtc` and `av`. Install with:\n\n"
            "```\npip install streamlit-webrtc av\n```"
        )
        return

    imgsz = settings["imgsz"]
    conf = settings["conf"]
    iou = settings["iou"]
    device = settings["device"]

    # Initialise live-stream credit tracking in session state
    for key in ("live_credit_start", "live_credits_deducted"):
        if key not in st.session_state:
            st.session_state[key] = None if key == "live_credit_start" else 0

    zoom = st.slider(
        "Zoom",
        min_value=1.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Digital zoom: center-crops the frame and rescales it before detection.",
    )

    class Transformer(VideoTransformerBase):
        def __init__(self) -> None:
            self.model = model
            self.names = names
            self.zoom = 1.0

        def transform(self, frame) -> np.ndarray:
            image_np = frame.to_ndarray(format="bgr24")
            if self.zoom > 1.0:
                h, w = image_np.shape[:2]
                new_h = max(1, int(h / self.zoom))
                new_w = max(1, int(w / self.zoom))
                y0 = (h - new_h) // 2
                x0 = (w - new_w) // 2
                cropped = image_np[y0:y0 + new_h, x0:x0 + new_w]
                pil_zoomed = Image.fromarray(cropped[:, :, ::-1]).resize((w, h), Image.BILINEAR)
                image_np = np.asarray(pil_zoomed)[:, :, ::-1]
            rgb = image_np[:, :, ::-1]
            predict_kwargs: dict[str, Any] = {
                "source": rgb,
                "imgsz": imgsz,
                "conf": conf,
                "iou": iou,
                "verbose": False,
                "save": False,
            }
            if device:
                predict_kwargs["device"] = device
            results = self.model.predict(**predict_kwargs)
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return image_np
            pil = Image.fromarray(rgb)
            annotated = draw_detections(
                pil,
                [tuple(float(v) for v in box) for box in boxes.xyxy.cpu().tolist()],
                [float(c) for c in boxes.conf.cpu().tolist()],
                [int(c) for c in boxes.cls.cpu().tolist()],
                self.names,
            )
            return np.asarray(annotated)[:, :, ::-1]

    webrtc_ctx = webrtc_streamer(
        key="txl-pbc-live",
        video_transformer_factory=Transformer,
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_transformer is not None:
        webrtc_ctx.video_transformer.zoom = zoom

    # Deduct credits based on elapsed wall-clock time (runs on each Streamlit rerun)
    if webrtc_ctx.state.playing:
        now = time.time()
        if st.session_state.live_credit_start is None:
            st.session_state.live_credit_start = now
            st.session_state.live_credits_deducted = 0
        else:
            elapsed = now - st.session_state.live_credit_start
            to_deduct = int(elapsed) - st.session_state.live_credits_deducted
            if to_deduct > 0:
                if _deduct_credits(to_deduct):
                    st.session_state.live_credits_deducted += to_deduct
                else:
                    st.error("Out of credits! Please stop the stream.")

        elapsed_total = time.time() - st.session_state.live_credit_start
        remaining = st.session_state.get("credits") or 0
        st.info(
            f"🔴 Streaming · {int(elapsed_total)}s elapsed · "
            f"{st.session_state.live_credits_deducted} credits used · "
            f"{remaining} credits remaining"
        )
    else:
        if st.session_state.live_credit_start is not None:
            # Stream stopped — reset tracking
            st.session_state.live_credit_start = None
            st.session_state.live_credits_deducted = 0


# ---------------------------------------------------------------------------
# Admin panel
# ---------------------------------------------------------------------------

def render_admin_panel() -> None:
    if st.button("← Back to app", key="admin_back"):
        st.session_state.show_admin = False
        st.rerun()

    st.title("Admin Panel")
    st.caption("Manage users and credits.")

    try:
        _ = st.secrets["firebase_admin"]
    except KeyError:
        st.error("Firebase Admin SDK is not configured in Streamlit secrets.")
        st.info(
            "Add a `[firebase_admin]` section to your Streamlit secrets "
            "with your Firebase service account JSON. Get it from:\n\n"
            "Firebase Console → Project Settings → Service accounts → "
            "**Generate new private key**"
        )
        st.code(
            '[firebase_admin]\ntype = "service_account"\nproject_id = "your-project-id"\n'
            'private_key_id = "..."\nprivate_key = "-----BEGIN RSA PRIVATE KEY-----\\n..."\n'
            'client_email = "firebase-adminsdk-xxx@your-project.iam.gserviceaccount.com"\n'
            'client_id = "..."\nauth_uri = "https://accounts.google.com/o/oauth2/auth"\n'
            'token_uri = "https://oauth2.googleapis.com/token"',
            language="toml",
        )
        return

    with st.spinner("Loading users…"):
        try:
            users = _admin_list_users()
        except Exception as exc:
            st.error(f"Failed to load users: {exc}")
            return

    # ---- Summary metrics ----
    col1, col2 = st.columns(2)
    col1.metric("Total users", len(users))
    col2.metric("Total credits in circulation", sum(u.get("credits", 0) for u in users))

    st.divider()

    # ---- Users table ----
    st.subheader("Users")
    search = st.text_input("Filter by email", placeholder="user@example.com")
    filtered = [u for u in users if search.lower() in u["email"].lower()] if search else users

    admin_emails = _get_admin_emails()
    display_rows = [
        {
            "Email": u["email"],
            "Credits": u.get("credits", 0),
            "Admin": "✓" if u["email"] in admin_emails else "",
            "UID": u["uid"],
        }
        for u in sorted(filtered, key=lambda x: x["email"])
    ]
    st.dataframe(display_rows, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Credit editor ----
    st.subheader("Edit Credits")
    all_emails = sorted(u["email"] for u in users if u["email"])

    if not all_emails:
        st.info("No users yet.")
        return

    with st.form("admin_credit_form"):
        selected_email = st.selectbox("User", options=all_emails)
        action = st.radio(
            "Action", ["Set to", "Add", "Subtract"], horizontal=True
        )
        amount = st.number_input(
            "Credits", min_value=0, max_value=100_000, value=10, step=1
        )
        submitted = st.form_submit_button("Apply", type="primary")

    if submitted:
        match = next((u for u in users if u["email"] == selected_email), None)
        if match:
            current = match.get("credits", 0)
            if action == "Set to":
                new_credits = int(amount)
            elif action == "Add":
                new_credits = current + int(amount)
            else:
                new_credits = max(0, current - int(amount))
            try:
                _admin_set_user_credits(match["uid"], new_credits)
                current_user = st.session_state.get("user") or {}
                if current_user.get("localId") == match["uid"]:
                    st.session_state.credits = new_credits
                st.success(
                    f"Updated **{selected_email}**: {current} → {new_credits} credits"
                )
            except Exception as exc:
                st.error(f"Failed to update credits: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="TXL-PBC YOLO26 Detector",
        page_icon=":microscope:",
        layout="wide",
    )

    _init_session()
    _restore_login_from_cookie()

    if not st.session_state.user:
        render_auth_page()
        st.stop()

    # Load credits once per session after login
    if st.session_state.get("credits") is None:
        _refresh_credits()

    _check_stripe_payment()

    # ---- Sidebar ----
    st.sidebar.divider()
    user_email = st.session_state.user.get("email", "")
    st.sidebar.caption(f"Signed in as **{user_email}**")

    credits = st.session_state.get("credits") or 0
    st.sidebar.metric("Credits", credits)

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Refresh", use_container_width=True, key="refresh_credits"):
        _refresh_credits()
        st.rerun()
    if col_b.button("Logout", use_container_width=True, key="logout_btn"):
        _do_logout()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Buy Credits")
    if st.sidebar.button("Buy 100 Credits ($5)", use_container_width=True, type="primary"):
        _create_stripe_checkout(100, 500)
    if st.sidebar.button("Buy 500 Credits ($20)"):
        _create_stripe_checkout(500, 2000)

    if _is_admin():
        st.sidebar.divider()
        label = "← Back to App" if st.session_state.get("show_admin") else "Admin Panel"
        if st.sidebar.button(label, use_container_width=True, key="admin_toggle"):
            st.session_state.show_admin = not st.session_state.get("show_admin", False)
            st.rerun()

    # ---- Admin panel takes over the page ----
    if st.session_state.get("show_admin") and _is_admin():
        render_admin_panel()
        st.stop()

    # ---- Header ----
    st.title("TXL-PBC YOLO26 Blood-Cell Detector")
    st.caption(
        "Real-time WBC / RBC / Platelet detection powered by the fine-tuned "
        "`txl_pbc_yolo26m2` checkpoint."
    )

    settings = sidebar_controls()

    try:
        model_path = resolve_model_path(settings["model_path"])
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.success(f"Using checkpoint:\n`{model_path}`")

    try:
        model = load_model(str(model_path))
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    names = class_names_from_model(model)

    mode = st.radio(
        "Input source",
        options=("Camera snapshot", "Upload image", "Batch images", "Upload video", "Live webcam"),
        horizontal=True,
    )

    if mode == "Camera snapshot":
        mode_camera(model, names, settings)
    elif mode == "Upload image":
        mode_upload_image(model, names, settings)
    elif mode == "Batch images":
        mode_batch_images(model, names, settings)
    elif mode == "Upload video":
        mode_upload_video(model, names, settings)
    else:
        mode_live(model, names, settings)


if __name__ == "__main__":
    main()
