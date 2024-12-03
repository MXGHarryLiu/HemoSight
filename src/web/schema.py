from typing import Optional
from typing_extensions import Annotated
from datetime import datetime
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictBytes
from pydantic.networks import EmailStr

PyObjectId = Annotated[str, BeforeValidator(str)]

class QueryImageModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    project_id: str = Field(...)
    filename: str = Field(...)
    content: StrictBytes = Field(...)
    model_config = ConfigDict(populate_by_name=True)

class ProjectModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: str = Field(...)
    name: str = Field(...)
    created_at: datetime = Field(...)
    status: str = Field(...)
    model_config = ConfigDict(populate_by_name=True)

class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    username: str = Field(..., max_length=64)
    email: EmailStr = Field(...)
    hashed_password: str = Field(...)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)
    model_config = ConfigDict(populate_by_name=True)

class CreateUserModel(BaseModel):
    username: str = Field(..., max_length=64)
    email: EmailStr = Field(...)
    password: str = Field(...)

class JobModal(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    type: str = Field(...)
    data: dict = Field(...)
    status: str = Field(...)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)
    model_config = ConfigDict(populate_by_name=True)
