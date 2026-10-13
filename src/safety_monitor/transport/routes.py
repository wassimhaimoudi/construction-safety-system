"""HTTP routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from ..application import analyze_frame
from ..application.detector import Detector
from ..config import Settings
from ..domain import taxonomy as tax
from ..domain import scoring
from ..domain.levels import CRITICAL, HIGH, LOW, MEDIUM, SAFE
from ..infrastructure.annotate import annotate
from ..infrastructure.codecs import InvalidImageError, decode_image
from ..infrastructure.person_crop import crop_persons
from .encoding import to_data_url
from .schemas import (
    AnalysisResponse,
    CropSchema,
    ErrorResponse,
    HealthResponse,
    InfoResponse,
)

router = APIRouter()


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_detector(request: Request) -> Detector:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="models are not loaded",
        )
    return detector


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    detector = getattr(request.app.state, "detector", None)
    return HealthResponse(
        status="ok" if detector is not None else "degraded",
        models_loaded=detector is not None,
        models=[settings.ppe_model_path.name, settings.scaffold_model_path.name],
    )


@router.get("/api/v1/info", response_model=InfoResponse, tags=["meta"])
def info() -> InfoResponse:
    return InfoResponse(
        ppe_classes={
            "helmet_present": sorted(tax.HELMET_PRESENT_CLASSES),
            "helmet_missing": sorted(tax.HELMET_MISSING_CLASSES),
            "vest_present": sorted(tax.VEST_PRESENT_CLASSES),
            "vest_missing": sorted(tax.VEST_MISSING_CLASSES),
            "glove_present": sorted(tax.GLOVE_PRESENT_CLASSES),
            "glove_missing": sorted(tax.GLOVE_MISSING_CLASSES),
            "mask_present": sorted(tax.MASK_PRESENT_CLASSES),
            "mask_missing": sorted(tax.MASK_MISSING_CLASSES),
            "anatomy": sorted(tax.ANATOMY_CLASSES),
        },
        scaffold_classes=sorted(tax.SCAFFOLD_CLASSES),
        infra_classes=sorted(tax.INFRA_CLASSES),
        ppe_weights=scoring.PPE_WEIGHTS,
        level_bands={
            SAFE: [81, 100],
            LOW: [51, 80],
            MEDIUM: [21, 50],
            HIGH: [1, 20],
            CRITICAL: [0, 0],
        },
        caps={
            "ppe": scoring.CAP_PPE,
            "scaffold": scoring.CAP_SCAFFOLD,
            "crowd": scoring.CAP_CROWD,
            "person": scoring.MAX_PERSON_SCORE,
        },
    )


@router.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    tags=["analysis"],
)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    draw: bool = Query(True, description="include the annotated image"),
    crops: bool = Query(True, description="include per-person image crops"),
    detector: Detector = Depends(get_detector),
    settings: Settings = Depends(get_settings),
) -> AnalysisResponse:
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"image exceeds {settings.max_upload_bytes} bytes",
        )

    try:
        frame = decode_image(payload)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    analysis = analyze_frame(detector, frame)

    annotated_image = (
        to_data_url(annotate(frame, analysis)) if draw and settings.annotate else None
    )
    crop_schemas = (
        [
            CropSchema(index=index, image=to_data_url(crop))
            for index, crop in crop_persons(frame, analysis.detections)
        ]
        if crops
        else []
    )

    return AnalysisResponse.from_domain(
        analysis.result,
        analysis.detections,
        annotated_image=annotated_image,
        crops=crop_schemas,
    )
