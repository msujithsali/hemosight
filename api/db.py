"""SQLAlchemy 2.x setup — SQLite on edge, Postgres on the aggregator.

Author: M Sujith Sali, ISE Dept, VTU Karnataka.
"""
from __future__ import annotations

import os

from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.environ.get("HEMOSIGHT_DB_URL", "sqlite:///hemosight_edge.db")
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    analysis_id: Mapped[str] = mapped_column(String, primary_key=True)
    provenance: Mapped[str] = mapped_column(String)
    model_version: Mapped[str] = mapped_column(String)
    mlflow_run_id: Mapped[str] = mapped_column(String)


def init_db() -> None:
    Base.metadata.create_all(engine)
