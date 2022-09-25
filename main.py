from typing import Union
import uuid
from fastapi import FastAPI, File, UploadFile
import os
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import dotenv_values
from aleph_alpha_client import AlephAlphaClient, AlephAlphaModel, QaRequest, Document, ImagePrompt
import base64
from logging.config import dictConfig
import logging
from config import LogConfig
from detection import get_classes_in_image, detect_video

dictConfig(LogConfig().dict())
logger = logging.getLogger("client")
config = dotenv_values(".env")
app = FastAPI(debug=True)

# create tmp folder.
os.makedirs("tmp_raw", exist_ok=True)
os.makedirs("tmp_processed", exist_ok=True)
os.makedirs("tmp_dict", exist_ok=True)

logger.info("Starting Backend Service")


class NLPRequest(BaseModel):
    question: str
    memory: str


class MultimodalRequest(BaseModel):
    img: str
    question: str

@app.get("/")
def read_root():
    return "Welcome to the Simple Aleph Alpha FastAPI Backend!"


# @app.get("/token")
# def return_token():
#     return config["token"]

@app.post("/nlp")
def nlp(request: NLPRequest):
    logger.info("Starting NLP Request")
    # sent request to aleph alpha
    model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=config["token"]),
    # You need to choose a model with qa support for this example.
    model_name = "luminous-extended")

    # important to remove all of the " and '. Otherwise the request will fail
    question = request.question.replace('"', '').replace("'", "")
    memory = request.memory.replace('"', '').replace("'", "")
    
    document = Document.from_text(memory)

    request = QaRequest(
        query = question,
        documents = [document],
    )

    result = model.qa(request)

    return result

@app.post("/multimodal")
async def multimodal(request: MultimodalRequest):
    logger.info("Starting Multimodal Request")
    
    model = AlephAlphaModel(
    AlephAlphaClient(host="https://api.aleph-alpha.com", token=config["token"]),
    # You need to choose a model with qa support for this example.
    model_name = "luminous-extended")
    img = str.encode(request.img)
    img = ImagePrompt.from_bytes(img)
    prompt = [img]
    document = Document.from_prompt(prompt)
    logger.info("Convertion sucessful.")
    request = QaRequest (
        query = request.question,
        documents = [document]
)   
    logger.info("Sending Request to Aleph Alpha.")
    result = model.qa(request)
    logger.info("Request sucessful.", result)

    print(result)
    return result

# object detection request.
@app.post("/detection_video")
async def detection_video(file: UploadFile):
    # get the video and save it locally.
    logger.info("Saving file locally.")
    id = str(uuid.uuid4())
    file_path = f"tmp_raw/{id}.{file.filename.split('.')[-1]}"
    save_path = f"tmp_processed/{id}.{file.filename.split('.')[-1]}"
    dict_path = f"tmp_dict/{id}.json"

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    # call the service to detect the video.
    try:
        detect_video(file_path, save_path, dict_path)
        logger.info("Video processed.")

    except Exception as e:
        logger.error("Error processing video.", e)
        return {"error": "Error processing video."}

    # return the video id.
    return id

# get detected video.
@app.get("/detection_video/{id}")
async def get_detected_video(id: str):
    return FileResponse(f"tmp_processed/{id}.mp4")

# get detected classes.
@app.get("/detection_classes/{id}")
async def get_detected_classes(id: str):
    with open(f"tmp_dict/{id}.json", "r") as f:
        return f.read()


