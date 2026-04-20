from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
from PIL import Image
import numpy as np
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

        embeddings = DeepFace.represent(
            img_path=image,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )

        if not embeddings or len(embeddings) == 0:
            return {"error": "No face detected"}

        if len(embeddings) > 1:
            return {"error": "Multiple faces detected. Only one allowed"}

        embedding = embeddings[0]["embedding"]

        return {"embedding": embedding}

    except Exception as e:
        print("Face API Error:", e)
        return {"error": "Processing failed"}
