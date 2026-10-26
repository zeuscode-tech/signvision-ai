from pydantic import BaseModel, Field, field_validator


class GestureCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=2000)
    is_unknown: bool = False

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Название не может быть пустым")
        return value


class SessionCreate(BaseModel):
    gesture_class_id: int
    note: str = Field(default="", max_length=200)


class LandmarkPayload(BaseModel):
    # 126 values: Left xyz (63), then Right xyz (63); missing hand is zeros.
    landmarks: list[float]


class FrameSequencePayload(BaseModel):
    # JPEG data URLs sampled from the local webcam. Frames are processed only
    # by the loopback FastAPI server and are never persisted.
    frames: list[str] = Field(min_length=1, max_length=16)


class SampleCreate(LandmarkPayload):
    session_id: int


class AttemptCreate(BaseModel):
    gesture_name: str
    mode: str = "practice"
    success: bool
    confidence: float | None = None
    duration_seconds: int = Field(default=0, ge=0)


class ImportPayload(BaseModel):
    attempts: list[dict] = []
    reviews: list[dict] = []

