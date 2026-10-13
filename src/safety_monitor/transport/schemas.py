"""Pydantic transport models (request/response shapes).

Domain objects are mapped here at the edge; the domain itself never imports
pydantic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..domain import Detection, PersonAssessment, SafetyResult


class BBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_tuple(cls, box: tuple[int, int, int, int]) -> "BBoxSchema":
        return cls(x1=box[0], y1=box[1], x2=box[2], y2=box[3])


class DetectionSchema(BaseModel):
    class_name: str
    confidence: float
    bbox: BBoxSchema

    @classmethod
    def from_domain(cls, det: Detection) -> "DetectionSchema":
        return cls(
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=BBoxSchema.from_tuple(det.bbox),
        )


class PersonSchema(BaseModel):
    index: int
    bbox: BBoxSchema
    on_scaffold: bool
    helmet: str
    vest: str
    glove: str
    mask: str
    status: str
    score: int
    reasons: list[str]

    @classmethod
    def from_domain(cls, person: PersonAssessment) -> "PersonSchema":
        return cls(
            index=person.index,
            bbox=BBoxSchema.from_tuple(person.bbox),
            on_scaffold=person.on_scaffold,
            helmet=person.helmet,
            vest=person.vest,
            glove=person.glove,
            mask=person.mask,
            status=person.status,
            score=person.score,
            reasons=list(person.reasons),
        )


class CropSchema(BaseModel):
    """A cropped person image, base64-encoded as a data URL."""

    index: int
    image: str


class AnalysisCounts(BaseModel):
    persons: int
    violations: int
    detections: int


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    safety: int = Field(ge=0, le=100)
    danger: int = Field(ge=0, le=100)
    level: str
    ppe_risk: int
    scaffold_risk: int
    crowd_risk: int
    counts: AnalysisCounts
    persons: list[PersonSchema]
    detections: list[DetectionSchema]
    annotated_image: str | None = None
    crops: list[CropSchema] = Field(default_factory=list)

    @classmethod
    def from_domain(
        cls,
        result: SafetyResult,
        detections: tuple[Detection, ...],
        *,
        annotated_image: str | None = None,
        crops: list[CropSchema] | None = None,
    ) -> "AnalysisResponse":
        return cls(
            safety=result.safety,
            danger=result.danger,
            level=result.level,
            ppe_risk=result.ppe_risk,
            scaffold_risk=result.scaffold_risk,
            crowd_risk=result.crowd_risk,
            counts=AnalysisCounts(
                persons=len(result.persons),
                violations=sum(1 for p in result.persons if p.status == "VIOLATION"),
                detections=len(detections),
            ),
            persons=[PersonSchema.from_domain(p) for p in result.persons],
            detections=[DetectionSchema.from_domain(d) for d in detections],
            annotated_image=annotated_image,
            crops=crops or [],
        )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    models_loaded: bool
    models: list[str]


class InfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    ppe_classes: dict[str, list[str]]
    scaffold_classes: list[str]
    infra_classes: list[str]
    ppe_weights: dict[str, dict[str, int]]
    level_bands: dict[str, list[int]]
    caps: dict[str, int]


class ErrorResponse(BaseModel):
    detail: str
