from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, EmailStr


class Upload(BaseModel):
    filename: str = Field(..., min_length=1)
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    file_type: str = Field(..., min_length=1)
    status: str = Field(default="uploaded")
    metadata: Optional[dict] = None


class User(BaseModel):
    email: EmailStr
    hashed_password: str
    created_date: datetime = Field(default_factory=datetime.utcnow)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class Upload(BaseModel):
    filename: str = Field(..., min_length=1)
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    file_type: str = Field(..., min_length=1)
    status: str = Field(default="uploaded")
    metadata: Optional[dict] = None
    user_id: str