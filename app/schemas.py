# app/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, Dict

class PromptConfig(BaseModel):
    """
    Configuration for document processing and response generation.
    """
    company_name: str = Field(
        default="Esaa",
        description="Company name for response context"
    )
    agent_role: str = Field(
        default="customer support agent",
        description="Role the AI should assume"
    )
    response_style: str = Field(
        description="Style guide for response generation"
    )
    tone: str = Field(
        description="Tone of the responses"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "TechCorp",
                "agent_role": "technical advisor",
                "response_style": "concise and technical",
                "tone": "professional"
            }
        }