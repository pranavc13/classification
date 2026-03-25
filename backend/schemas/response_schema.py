from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    classification: str
    features: list[str]
    description: str
    improvements: list[str]
    generated_image: str = Field(..., description="Base64 encoded generated image")


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
