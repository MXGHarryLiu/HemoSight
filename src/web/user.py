import logging
from typing import Union
from typing_extensions import Annotated
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
# custom module
from web.schema import UserModel, CreateUserModel
from web.database import get_database

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logger = logging.getLogger('uvicorn')
logger.setLevel(logging.DEBUG)

# database
db = get_database()
user_collection = db['user']

# openssl rand -hex 32 to generate a secret key
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SECRET_KEY = "e6529ecc336cad0390e287a0b2203b13b4f80c2c7be7dbb62ecc57dafb266b5d"
ALGORITHM = "HS256"

class Token(BaseModel):
    access_token: str
    token_type: str

async def get_user(username: str) -> Union[UserModel, None]:
    user_entry = await user_collection.find_one({"username": username})
    if user_entry is None:
        return None
    return UserModel(**user_entry)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

async def authenticate_user(username: str, password: str) -> Union[UserModel, None]:
    user = await get_user(username=username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserModel:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try: 
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await get_user(username=username)
    if user is None:
        raise credentials_exception
    return user

@router.post("/token", response_class=JSONResponse, 
             response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, 
                            detail="Incorrect username or password", 
                            headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@router.get("/current_user", response_class=JSONResponse, 
            response_model=UserModel)
async def current_user(user: Annotated[dict, Depends(get_current_user)]):
    return user

@router.post("/user", response_class=JSONResponse)
async def create_user(user: CreateUserModel):
    # check duplicate username
    if await user_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    # check duplicate email
    if await user_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already exists")
    hashed_password = pwd_context.hash(user.password)
    user_entry = UserModel(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )
    new_user = await user_collection.insert_one(
        user_entry.model_dump(by_alias=True, exclude=["id"]))
    id = str(new_user.inserted_id)
    return {"message": "User created successfully", "user_id": id}
