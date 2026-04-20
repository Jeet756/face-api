from fastapi import FastAPI, File, UploadFile
import face_recognition
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.post("/get-embedding")
async def get_embedding(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if len(contents) > 2 * 1024 * 1024:
            return {"error": "Image too large"}

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)

        locations = face_recognition.face_locations(image)

        if len(locations) == 0:
            return {"error": "No face detected"}

        if len(locations) > 1:
            return {"error": "Multiple faces detected"}

        encodings = face_recognition.face_encodings(image, locations)

        return {"embedding": encodings[0].tolist()}

    except Exception as e:
        print("Error:", e)
        return {"error": "Processing failed"}
