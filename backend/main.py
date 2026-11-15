from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.detector_service import YOLODetectorService
from backend.schemas import DetectionResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = YOLODetectorService()
    yield


app = FastAPI(
    title="YOLOv2 Vision Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_detector(app_instance: FastAPI) -> YOLODetectorService:
    detector = getattr(app_instance.state, "detector", None)
    if detector is None:
        detector = YOLODetectorService()
        app_instance.state.detector = detector
    return detector


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    detector = _get_detector(app)
    return HealthResponse(
        status="ok",
        device=str(detector.device),
        model_loaded=True,
        num_classes=detector.num_classes,
        img_size=detector.default_img_size,
        classes=detector.classes,
    )


@app.post("/predict/image", response_model=DetectionResponse)
async def predict_image(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.01, le=0.99),
    iou_threshold: float = Query(0.45, ge=0.05, le=0.95),
    img_size: int = Query(416, ge=128, le=960),
) -> DetectionResponse:
    detector = _get_detector(app)
    image_bytes = await file.read()
    try:
        return await asyncio.to_thread(
            detector.predict_image_bytes,
            image_bytes,
            conf_threshold,
            iou_threshold,
            img_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}") from exc


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket) -> None:
    await websocket.accept()
    detector = _get_detector(app)
    defaults = detector.config.get("defaults", {})
    conf_threshold = float(defaults.get("conf_threshold", 0.25))
    iou_threshold = float(defaults.get("iou_threshold", 0.45))
    img_size = int(detector.default_img_size)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_data = message.get("text")
            if text_data is not None:
                try:
                    payload = json.loads(text_data)
                    if isinstance(payload, dict) and payload.get("type") == "config":
                        if "conf_threshold" in payload:
                            conf_threshold = max(0.01, min(0.99, float(payload["conf_threshold"])))
                        if "iou_threshold" in payload:
                            iou_threshold = max(0.05, min(0.95, float(payload["iou_threshold"])))
                        if "img_size" in payload:
                            img_size = int(payload["img_size"])
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    await websocket.send_json({"error": f"Invalid config payload: {exc}"})
                continue

            frame_bytes = message.get("bytes")
            if frame_bytes is not None:
                try:
                    result = await asyncio.to_thread(
                        detector.predict_image_bytes,
                        frame_bytes,
                        conf_threshold,
                        iou_threshold,
                        img_size,
                    )
                    await websocket.send_json(result.model_dump())
                except Exception as exc:
                    await websocket.send_json({"error": f"Frame inference error: {exc}"})
    except WebSocketDisconnect:
        return

