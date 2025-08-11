from fastapi import FastAPI, Depends, Form, UploadFile, File
from app.dependencies import get_distance_service
from app.services.distance_service import DistanceService
import tempfile
import shutil

app = FastAPI()


@app.post("/predict", response_model=dict)
async def predict_distance(
    file: UploadFile = File(...),
    camera_pov: float = Form(...),
    height: float = Form(...),
    obj_class: str = Form(...),
    service: DistanceService = Depends(get_distance_service)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    results = service.process_image(tmp_path, camera_pov, height, obj_class)
    return {"results": results}
