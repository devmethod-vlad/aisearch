from pydantic import BaseModel


class HealthCheckResult(BaseModel):
    status: str
    message: str


class HealthCheckResponse(BaseModel):
    status: str
    services: dict[str, HealthCheckResult]
