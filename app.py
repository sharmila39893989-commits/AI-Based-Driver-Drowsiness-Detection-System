from pathlib import Path
from typing import Union

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

if __package__:
    from .src.alert_system import should_alert
    from .src.drowsiness_detection import detect_drowsiness
    from .src.drowsiness_engine import calculate_risk_score
    from .src.eye_detection import detect_eye
    from .src.head_pose import detect_driver_features
    from .src.yawn_detection import detect_yawn
else:
    from src.alert_system import should_alert
    from src.drowsiness_detection import detect_drowsiness
    from src.drowsiness_engine import calculate_risk_score
    from src.eye_detection import detect_eye
    from src.head_pose import detect_driver_features
    from src.yawn_detection import detect_yawn


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="JAAGRATA AI Driver Drowsiness Detection API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
REACT_BUILD_DIR = STATIC_DIR / "react-app"


def serve_frontend_page() -> Union[FileResponse, HTMLResponse]:
    react_index = REACT_BUILD_DIR / "index.html"
    if react_index.is_file():
        return FileResponse(react_index)

    static_index = STATIC_DIR / "index.html"
    if static_index.is_file():
        return FileResponse(static_index)

    return HTMLResponse(
        """
        <!doctype html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\" />
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
            <title>JAAGRATA</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    display: grid;
                    place-items: center;
                    min-height: 100vh;
                    background: #0f172a;
                    color: #e2e8f0;
                }
                .card {
                    background: rgba(15, 23, 42, 0.9);
                    padding: 2rem 3rem;
                    border-radius: 14px;
                    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
                    text-align: center;
                }
                h1 {
                    margin-bottom: 0.5rem;
                }
                p {
                    margin: 0;
                    color: #cbd5e1;
                }
            </style>
        </head>
        <body>
            <div class=\"card\">
                <h1>JAAGRATA</h1>
                <p>Driver drowsiness monitoring is ready.</p>
            </div>
        </body>
        </html>
        """,
        media_type="text/html",
    )


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return serve_frontend_page()


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# IMAGE DECODER
# ============================================================

def decode_image(image_bytes):

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    return frame


# ============================================================
# EYE RESULT COMBINATION
# ============================================================

def combine_eye_results(
    left_status,
    left_confidence,
    right_status,
    right_confidence
):

    left_closed_confidence = (
        left_confidence
        if left_status == "Closed"
        else 1 - left_confidence
    )
    right_closed_confidence = (
        right_confidence
        if right_status == "Closed"
        else 1 - right_confidence
    )

    closed_confidence = (
        left_closed_confidence
        * right_closed_confidence
    )
    status = (
        "Closed"
        if left_status == "Closed" and right_status == "Closed"
        else "Open"
    )
    confidence = (
        closed_confidence
        if status == "Closed"
        else 1 - closed_confidence
    )

    return (
        status,
        round(float(confidence), 3)
    )


# ============================================================
# DETECTION API
# ============================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    live_detection: bool = Form(False)
):

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            return {
                "success": False,
                "error": "The uploaded file is empty. Please choose a non-empty image."
            }

        frame = decode_image(
            image_bytes
        )

        if frame is None:

            return {
                "success": False,
                "error": "Could not decode the uploaded file as an image. Please choose a valid image."
            }

        source_frame = frame.copy()

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        (
            _,
            head_status,
            _,
            eye_regions,
            face_image
        ) = await run_in_threadpool(
            detect_driver_features,
            source_frame.copy()
        )

        if face_image is None:
            return {
                "success": True,
                "face_detected": False,
                "eye": {
                    "status": "Not detected",
                    "confidence": 0.0
                },
                "yawn": {
                    "status": "Not detected",
                    "confidence": 0.0
                },
                "drowsiness": {
                    "status": "Not detected",
                    "confidence": 0.0
                },
                "head_pose": head_status,
                "risk_score": None,
                "risk_score_without_yawn": None,
                "safety_score": None,
                "safety_status": "NO FACE",
                "alert": False
            }


        # ----------------------------------------------------
        # DEFAULT VALUES
        # ----------------------------------------------------

        eye_status = "Not detected"
        eye_confidence = 0.0

        yawn_status = "Not detected"
        yawn_confidence = 0.0

        drowsiness_status = "Not detected"
        drowsiness_confidence = 0.0


        # ----------------------------------------------------
        # IF FACE FOUND
        # ----------------------------------------------------

        if face_image is not None:

            # ------------------------------------------------
            # YAWN DETECTION
            # ------------------------------------------------

            yawn_status, yawn_confidence = (
                await run_in_threadpool(
                    detect_yawn,
                    source_frame
                )
            )


            # ------------------------------------------------
            # DROWSINESS DETECTION
            # ------------------------------------------------

            drowsiness_status, drowsiness_confidence = (
                await run_in_threadpool(
                    detect_drowsiness,
                    source_frame
                )
            )


            # ------------------------------------------------
            # EYE REGION DETECTION
            # ------------------------------------------------

            left_eye = eye_regions[0] if len(eye_regions) > 0 else None
            right_eye = eye_regions[1] if len(eye_regions) > 1 else None


            # ------------------------------------------------
            # LEFT + RIGHT EYE MODEL
            # ------------------------------------------------

            if (
                left_eye is not None
                and right_eye is not None
            ):

                left_status, left_confidence = (
                    await run_in_threadpool(
                        detect_eye,
                        left_eye
                    )
                )

                right_status, right_confidence = (
                    await run_in_threadpool(
                        detect_eye,
                        right_eye
                    )
                )

                (
                    eye_status,
                    eye_confidence
                ) = combine_eye_results(
                    left_status,
                    left_confidence,
                    right_status,
                    right_confidence
                )

            elif left_eye is not None:

                eye_status, eye_confidence = (
                    await run_in_threadpool(
                        detect_eye,
                        left_eye
                    )
                )

            elif right_eye is not None:

                eye_status, eye_confidence = (
                    await run_in_threadpool(
                        detect_eye,
                        right_eye
                    )
                )


        # ----------------------------------------------------
        # RISK SCORE
        # ----------------------------------------------------

        risk_score, safety_status = (
            calculate_risk_score(
                eye_status,
                yawn_status,
                drowsiness_status,
                head_status
            )
        )
        risk_score_without_yawn, _ = calculate_risk_score(
            eye_status,
            "No Yawn",
            drowsiness_status,
            head_status
        )


        # ----------------------------------------------------
        # ALERT
        # ----------------------------------------------------

        alert = should_alert(
            eye_status,
            yawn_status,
            drowsiness_status,
            include_yawn=not live_detection
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "success": True,

            "face_detected":
                face_image is not None,

            "eye": {
                "status":
                    eye_status,

                "confidence":
                    eye_confidence
            },

            "yawn": {
                "status":
                    yawn_status,

                "confidence":
                    yawn_confidence
            },

            "drowsiness": {
                "status":
                    drowsiness_status,

                "confidence":
                    drowsiness_confidence
            },

            "head_pose":
                head_status,

            "risk_score":
                risk_score,

            "risk_score_without_yawn":
                risk_score_without_yawn,

            "safety_score":
                100 - risk_score,

            "safety_status":
                safety_status,

            "alert":
                alert
        }


    except Exception as e:

        print(
            "Detection API error:",
            e
        )

        return {

            "success": False,

            "error":
                str(e)
        }


@app.get("/{page_name}")
def frontend_page(page_name: str):
    if page_name in {
        "live",
        "analytics",
        "alerts",
        "history",
        "about",
    }:
        return serve_frontend_page()

    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/react-app/{asset_path:path}")
def frontend_asset(asset_path: str):
    build_root = REACT_BUILD_DIR.resolve()
    asset_file = (build_root / asset_path).resolve()

    try:
        asset_file.relative_to(build_root)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Asset not found") from error

    if not asset_file.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")

    return FileResponse(asset_file)