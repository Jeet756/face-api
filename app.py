from fastapi import FastAPI, File, UploadFile
import face_recognition
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.post("/get-embedding")
async def get_embedding(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            return {"error": "Only JPG and PNG images allowed"}

        contents = await file.read()

        if len(contents) > 2 * 1024 * 1024:
            return {"error": "Image too large"}

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(image)

        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            return {"error": "No face detected"}

        if len(face_locations) > 1:
            return {"error": "Multiple faces detected. Only one allowed"}

        encodings = face_recognition.face_encodings(image, face_locations)

        if len(encodings) == 0:
            return {"error": "No face detected"}

        if len(encodings) > 1:
            return {"error": "Multiple faces detected. Only one allowed"}

        embedding = encodings[0]

        if len(embedding) != 128:
            return {"error": "Invalid embedding generated"}

        return {"embedding": embedding.tolist()}

    except Exception as e:
        print("Face API Error:", e)
        return {"error": "Processing failed"}
