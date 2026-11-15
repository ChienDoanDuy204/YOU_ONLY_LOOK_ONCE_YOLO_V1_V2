from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    width: int
    height: int
    inference_ms: float
    detections: list[DetectionBox] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    num_classes: int
    img_size: int
    classes: list[str]

