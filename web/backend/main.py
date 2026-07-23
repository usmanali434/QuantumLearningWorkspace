import os
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware

from models import SignupRequest, LoginRequest, Upload
from database import get_users_collection, get_uploads_collection
from auth_utils import hash_password, verify_password, create_access_token, get_current_user_email

app = FastAPI(title="StudyMind AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "uploaded_files"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/signup")
async def signup(request: SignupRequest):
    users = get_users_collection()

    existing_user = await users.find_one({"email": request.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed = hash_password(request.password)

    new_user = {
        "email": request.email,
        "hashed_password": hashed,
    }

    await users.insert_one(new_user)

    return {"message": "User created successfully.", "email": request.email}


@app.post("/login")
async def login(request: LoginRequest):
    users = get_users_collection()

    user = await users.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(email=user["email"])

    return {"access_token": token, "token_type": "bearer"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user_email: str = Depends(get_current_user_email),
):
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    upload_record = Upload(
        filename=file.filename,
        file_type=file.content_type,
        user_id=current_user_email,
    )

    uploads = get_uploads_collection()
    await uploads.insert_one(upload_record.model_dump())

    return {
        "message": "File uploaded successfully.",
        "filename": upload_record.filename,
        "status": upload_record.status,
    }

@app.get("/uploads")
async def get_uploads(current_user_email: str = Depends(get_current_user_email)):
    uploads = get_uploads_collection()

    user_uploads = []
    cursor = uploads.find({"user_id": current_user_email})

    async for document in cursor:
        user_uploads.append({
            "filename": document["filename"],
            "upload_date": document["upload_date"],
            "file_type": document["file_type"],
            "status": document["status"],
        })

    return user_uploads