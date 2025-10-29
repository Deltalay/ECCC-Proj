import asyncio
import base64
import json
from fastapi import FastAPI, File, UploadFile, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from duke import dcmread_image
import numpy as np
from ultralytics import YOLO
import pydicom
import cv2

app = FastAPI()
api = APIRouter(prefix="/api/v1")
origins = ["http://localhost:5173"]
MODEL = YOLO("./model/best.pt")
print("MODEL TYPE:", type(MODEL))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(
    "WARNING: THIS MODEL IS FOR RESEARCH PURPOSES ONLY. "
    "DO NOT USE THIS FOR CLINICAL DIAGNOSIS OR COMMERCIAL PURPOSES. "
    "THE AUTHORS ARE NOT RESPONSIBLE FOR ANY HARM OR LOSSES CAUSED BY THIS MODEL."
)


def extract_laterality_from_sequences(ds):
    """Extract laterality from sequences if available (e.g., tomosynthesis)"""
    if hasattr(ds, "SharedFunctionalGroupsSequence"):
        try:
            shared_seq = ds.SharedFunctionalGroupsSequence[0]
            if hasattr(shared_seq, "FrameAnatomySequence"):
                frame_anatomy = shared_seq.FrameAnatomySequence[0]
                if hasattr(frame_anatomy, "FrameLaterality"):
                    return frame_anatomy.FrameLaterality
        except (IndexError, AttributeError):
            pass

    if hasattr(ds, "ViewCodeSequence"):
        try:
            for item in ds.ViewCodeSequence:
                code_meaning = str(item.get((0x0008, 0x0104), "")).lower()
                if "left" in code_meaning:
                    return "L"
                elif "right" in code_meaning:
                    return "R"
        except (AttributeError, IndexError):
            pass

    return None


def extract_standardized_laterality(ds):
    """Extract and standardize breast laterality to single character"""
    laterality = (
        ds.get((0x0020, 0x9072), None)  # Frame Laterality
        or ds.get((0x0020, 0x0062), None)  # Image Laterality
        or extract_laterality_from_sequences(ds)
        or "U"  # Unknown
    )

    laterality = str(laterality).upper().strip()
    if laterality in ["L", "LEFT"]:
        return "L"
    elif laterality in ["R", "RIGHT"]:
        return "R"
    elif laterality in ["B", "BILATERAL"]:
        return "B"
    else:
        return "U"


def extract_standardized_view_position(ds):
    """Extract and standardize view position"""
    element = ds.get((0x0018, 0x5101), None)
    if element is not None:
        # get the actual string value
        view_position = str(element.value).upper().strip()
    else:
        view_position = "UNK"

    # Standardize common view position codes
    view_mapping = {
        "CC": "CC",
        "MLO": "MLO",
        "ML": "ML",
        "LM": "LM",
        "AT": "AT",
        "FB": "FB",
        "XCCL": "XCCL",
        "CV": "CV",
        "SIO": "SIO",
        "LL": "LL",
        "LMO": "LMO",
        " MLO": "MLO",
        "MLO ": "MLO",
    }

    return view_mapping.get(view_position, view_position)


def get_standard_breast_view(ds, suffix=None):
    """Map DICOM laterality + view to standardized breast view string"""
    laterality = extract_standardized_laterality(ds)
    view_position = extract_standardized_view_position(ds)

    mapping = {
        ("L", "CC"): "LCC",
        ("R", "CC"): "RCC",
        ("L", "MLO"): "LMLO",
        ("R", "MLO"): "RMLO",
        ("L", "ML"): "LML",
        ("R", "ML"): "RML",
        ("L", "LM"): "LLM",
        ("R", "LM"): "RLM",
    }

    view = mapping.get((laterality, view_position),
                       f"{laterality}{view_position}")

    if suffix is not None:
        view = f"{view}{suffix}"

    return view


def gray_scale(ds):
    frames = ds.pixel_array
    process_frames = []
    if frames.ndim == 2:
        frames = np.expand_dims(frames, axis=0)
    for frame in frames:
        norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        gray_image = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2GRAY)
        process_frames.append(gray_image)
    return process_frames


"""
AGAIN! THIS MODEL IS FOR RESEARCH PURPOSE ONLY.
DO NOT USE THIS FOR CLINICAL DIAGNOSE OR COMMERICAL PURPOSE.
WE DO NOT HELD RESPONSIBLE FOR ANY HARM THAT MAY CAUSE BY THIS MODEL.
"""


def preprocess_frame(frame, imgsz=512):
    """
    Preprocess a grayscale frame for YOLO11 1-channel model:
      - Normalize to 0–1 float32
      - Resize with letterbox padding to imgsz×imgsz
      - Keep single channel only
    """
    frame = frame.astype(np.float32)
    frame /= frame.max() if frame.max() > 0 else 1.0

    h, w = frame.shape
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (imgsz - nh) // 2
    bottom = imgsz - nh - top
    left = (imgsz - nw) // 2
    right = imgsz - nw - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Add a channel dimension → (imgsz, imgsz, 1)
    padded = padded[..., None]
    return padded, scale, left, top


def inf_yolo(frame, model, conf_thresh=0.2, iou_thresh=0.45, imgsz=512):
    """
    Inference for a 1-channel YOLO11 model.
    """
    # img, scale, pad_x, pad_y = preprocess_frame(frame, imgsz)



    result = model.predict(
        source=frame,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        stream=False
    )

    boxes = []
    for r in result:
        if r.boxes is None:
            continue
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(float, box.tolist())
           
            boxes.append([x1, y1, x2, y2])
    return boxes


def draw_box(frame, detections, color=(0, 255, 0), thickness=2):
    """
    Draw colored boxes on grayscale image.
    """
    # Convert grayscale to BGR for colored boxes
    if len(frame.shape) == 2:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_bgr = frame.copy()

    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

    return frame_bgr

# --- Streaming route ---


@api.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .dcm files are allowed.",
        )

    try:
        # Read DICOM file
        # ds = pydicom.dcmread(file.file)

        # Optional: your breast view logic (if needed)
        # view = get_standard_breast_view(ds)
        # print(view)
        data = dcmread_image(file.file, "lcc")

        # Stream frames
        async def frame_generator():
            for idx, frame in enumerate(data):

                detections = inf_yolo(frame, MODEL)
                boxed = draw_box(frame, detections)
                _, buffer = cv2.imencode(".png", boxed)
                b64 = base64.b64encode(buffer).decode("utf-8")

                yield json.dumps({"index": idx, "image": b64}) + "\n"

                await asyncio.sleep(0.2)

        return StreamingResponse(
            frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )


app.include_router(router=api)
