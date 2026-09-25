from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )