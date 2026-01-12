from __future__ import annotations

from typing import Any, List
from pydantic import BaseModel, Field, AliasChoices

class SourceReference(BaseModel):
    pdf_sha1: str
    page_index: int  # zero-based page index

class Answer(BaseModel):
    value: Any
    references: List[SourceReference] = Field(default_factory=list)
    # optional fields accepted by the official evaluator and helpful for alignment
    question_text: str | None = None
    kind: str | None = None

class Submission(BaseModel):
    # Some deployments use 'email', others use 'team_email'.
    email: str = Field(validation_alias=AliasChoices("email", "team_email"))
    submission_name: str
    answers: List[Answer]
