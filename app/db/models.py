# app/db/models.py
from sqlalchemy import Column, String, JSON, DateTime, Text
from sqlalchemy.sql import func
from .session import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # UUID
    name = Column(String)
    prompt_config = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    