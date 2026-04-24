"""Streamlit app for TXL-PBC YOLO26 blood-cell detection.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Supabase auth
# ---------------------------------------------------------------------------

@st.cache_resource
def _supabase_client():
    from supabase import create_client
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


def _init_session() -> None:
    for key in ("user", "session", "auth_error"):
        if key not in st.session_state:
            st.session_state[key] = None


def _do_login(email: str, password: str) -> None:
    try:
        res = _supabase_client().auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        st.session_state.user = res.user
        st.session_state.session = res.session
        st.session_state.auth_error = None
    except Exception as exc:
        st.session_state.auth_error = str(exc)


def _do_register(email: str, password: str) -> None:
    try:
        res = _supabase_client().auth.sign_up(
            {"email": email, "password": password}
        )
        if res.user and res.session:
            st.session_state.user = res.user
            st.session_state.session = res.session
            st.session_state.auth_error = None
        else:
            st.session_state.auth_error = (
                "Account created — check your email to confirm before logging in."
            )
    except Exception as exc:
        st.session_state.auth_error = str(exc)


def _do_logout() -> None:
    try:
        _supabase_client().auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.session_state.session = None
    st.session_state.auth_error = None


def render_auth_page() -> None:
    st.title("TXL-PBC YOLO26 — Sign in")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            _do_login(email, password)
            st.rerun()

    with tab_register:
        with st.form("register_form"):
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password", type="password", key="reg_pw")
            password2 = st.text_input("Confirm password", type="password", key="reg_pw2")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            if password != password2:
                st.session_state.auth_error = "Passwords do not match."
            else:
                _do_register(email, password)
                st.rerun()

    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)

DEFAULT_MODEL_PATH = Path("runs/yolo26/txl_pbc_yolo26m2/weights/best.pt")
FALLBACK_MODEL_PATH = Path("runs/yolo26/txl_pbc_yolo26m/weights/best.pt")
CLASS_NAMES = ["WBC", "RBC", "Platelets"]
CLASS_COLORS = {
    "WBC": (22, 163, 74),
    "RBC": (220, 38, 38),
    "Platelets": (37, 99, 235),
}


@st.cache_resource(show_spinner="Loading YOLO26 checkpoint...")
def load_model(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc
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
        Path("runs").glob("**/weights/best.pt"),
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
            st.dataframe(rows, use_container_width=True)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def mode_camera(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Camera snapshot")
    st.caption(
        "Allow camera access in your browser, take a photo, and the model will "
        "annotate detections."
    )
    snapshot = st.camera_input("Take a photo")
    if snapshot is None:
        return
    image = Image.open(snapshot).convert("RGB")
    detections = run_inference(
        model,
        image,
        imgsz=settings["imgsz"],
        conf=settings["conf"],
        iou=settings["iou"],
        device=settings["device"],
    )
    annotated = draw_detections(
        image,
        detections["boxes"],
        detections["confs"],
        detections["classes"],
        names,
    )
    left, right = st.columns(2)
    left.image(image, caption="Snapshot", use_container_width=True)
    right.image(annotated, caption="Predictions", use_container_width=True)
    render_detection_summary(detections, names)
    st.download_button(
        "Download annotated image",
        data=image_to_png_bytes(annotated),
        file_name="prediction.png",
        mime="image/png",
    )


def mode_upload(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Upload image")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        return
    image = Image.open(uploaded).convert("RGB")
    detections = run_inference(
        model,
        image,
        imgsz=settings["imgsz"],
        conf=settings["conf"],
        iou=settings["iou"],
        device=settings["device"],
    )
    annotated = draw_detections(
        image,
        detections["boxes"],
        detections["confs"],
        detections["classes"],
        names,
    )
    left, right = st.columns(2)
    left.image(image, caption="Original", use_container_width=True)
    right.image(annotated, caption="Predictions", use_container_width=True)
    render_detection_summary(detections, names)
    st.download_button(
        "Download annotated image",
        data=image_to_png_bytes(annotated),
        file_name=f"{Path(uploaded.name).stem}_pred.png",
        mime="image/png",
    )


def mode_live(model: Any, names: list[str], settings: dict[str, Any]) -> None:
    st.subheader("Live webcam (WebRTC)")
    st.caption(
        "Streams frames from your browser camera through WebRTC and runs the "
        "model on each frame. Lower image size for smoother framerates."
    )
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

    class Transformer(VideoTransformerBase):
        def __init__(self) -> None:
            self.model = model
            self.names = names

        def transform(self, frame) -> np.ndarray:
            image_np = frame.to_ndarray(format="bgr24")
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

    webrtc_streamer(
        key="txl-pbc-live",
        video_transformer_factory=Transformer,
        media_stream_constraints={"video": True, "audio": False},
    )


def main() -> None:
    st.set_page_config(
        page_title="TXL-PBC YOLO26 Detector",
        page_icon=":microscope:",
        layout="wide",
    )

    _init_session()

    if not st.session_state.user:
        render_auth_page()
        st.stop()

    # Logout button in sidebar
    st.sidebar.divider()
    st.sidebar.caption(f"Signed in as **{st.session_state.user.email}**")
    if st.sidebar.button("Logout", use_container_width=True):
        _do_logout()
        st.rerun()

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
        options=("Camera snapshot", "Live webcam", "Upload image"),
        horizontal=True,
    )

    if mode == "Camera snapshot":
        mode_camera(model, names, settings)
    elif mode == "Live webcam":
        mode_live(model, names, settings)
    else:
        mode_upload(model, names, settings)


if __name__ == "__main__":
    main()
