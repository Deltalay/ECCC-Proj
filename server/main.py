from fastapi import FastAPI, File, UploadFile, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from duke import dcmread_image
from pydantic import BaseModel

app = FastAPI()
api = APIRouter(prefix="/api/v1")
origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Item(BaseModel):
    file: File


@api.post("/detect", status.HTTP_201_CREATED)
async def detect(file: UploadFile = File(...), ):
    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .dcm files are allowed."
        )
    try:
        images = dcmread_image(file.file)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
