"""JSON manifest of a ready-made recognition model.

The manifest is the single source of truth for preprocessing: every part of
the pipeline reads crop, resize, colour, normalisation and tensor layout from
here instead of hard-coding them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MODEL_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{1,63}$"


class Preprocessing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color: Literal["RGB", "BGR"] = "RGB"
    width: int = Field(ge=16, le=1024)
    height: int = Field(ge=16, le=1024)
    # stretch: plain resize (aspect ratio changes); pad: keep aspect ratio and letterbox;
    # center_crop: keep aspect ratio and crop the centre.
    resize: Literal["stretch", "pad", "center_crop"] = "stretch"
    pad_value: int = Field(default=114, ge=0, le=255)
    # none: whole frame; hands: square crop around detected hands with a safety margin.
    crop: Literal["none", "hands"] = "none"
    crop_margin: float = Field(default=0.35, ge=0, le=3)
    # scale_0_1: x/255; imagenet: (x - mean) / std on 0..255 values; custom: mean/std below; none: raw 0..255.
    normalization: Literal["scale_0_1", "imagenet", "custom", "none"]
    mean: list[float] | None = None
    std: list[float] | None = None
    # Axis order of the 5-D input tensor: N=batch, C=channels, T=time, H, W.
    layout: Literal["NCTHW", "NTCHW"] = "NCTHW"

    @model_validator(mode="after")
    def custom_needs_values(self) -> Preprocessing:
        if self.normalization == "custom":
            if not self.mean or not self.std or len(self.mean) != 3 or len(self.std) != 3:
                raise ValueError("normalization=custom требует mean и std из трёх значений")
            if any(value == 0 for value in self.std):
                raise ValueError("std не может содержать 0")
        return self


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str = Field(pattern=MODEL_ID_PATTERN)
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=40)
    type: Literal["isolated_sign"] = "isolated_sign"
    model_path: str = Field(min_length=1, max_length=300)
    labels_path: str = Field(min_length=1, max_length=300)
    input_shape: list[int] = Field(min_length=5, max_length=5)
    frame_count: int = Field(ge=1, le=256)
    # Values the model produces: raw logits (softmax applied here) or probabilities.
    output: Literal["logits", "probabilities"]
    # Label of the model's own "no gesture" class, if it has one.
    no_event_label: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    stability_count: int | None = Field(default=None, ge=1, le=20)
    cooldown_ms: int | None = Field(default=None, ge=0, le=60_000)
    # Run inference every N new frames once the window is full.
    inference_stride: int | None = Field(default=None, ge=1, le=256)
    providers: list[str] | None = None
    preprocessing: Preprocessing
    source: dict[str, Any] | None = None

    @field_validator("input_shape")
    @classmethod
    def positive_dims(cls, value: list[int]) -> list[int]:
        if any(dim < 1 for dim in value):
            raise ValueError("input_shape должен содержать положительные размерности")
        if value[0] != 1:
            raise ValueError("Поддерживается только batch = 1")
        return value

    @model_validator(mode="after")
    def shape_matches_preprocessing(self) -> ModelManifest:
        axes = dict(zip(self.preprocessing.layout, self.input_shape, strict=True))
        if axes["C"] != 3:
            raise ValueError("Ожидается 3 цветовых канала")
        if axes["T"] != self.frame_count:
            raise ValueError("frame_count не совпадает с временной осью input_shape")
        if axes["H"] != self.preprocessing.height or axes["W"] != self.preprocessing.width:
            raise ValueError("width/height preprocessing не совпадают с input_shape")
        return self

    @property
    def stride(self) -> int:
        return self.inference_stride or max(1, self.frame_count // 4)
