import json
import logging
import os
import uuid
import glob
from logging.config import dictConfig

from aleph_alpha_client import AlephAlphaClient, AlephAlphaModel, Document, ImagePrompt, QaRequest
from dotenv import dotenv_values
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

from app.config import LogConfig
from app.detection import detect_video, detect_single_image, speech_to_text, initialize_models

# initialize the Logging.
dictConfig(LogConfig().dict())
logger = logging.getLogger("client")
config = dotenv_values(".env")

# initialize the Fast API Application.
app = FastAPI(debug=True)

# create tmp folder.
os.makedirs("tmp_raw", exist_ok=True)
os.makedirs("tmp_img", exist_ok=True)
os.makedirs("tmp_processed", exist_ok=True)
os.makedirs("tmp_dict", exist_ok=True)

logger.info("Starting Backend Service")

# initialize the models from the hub.
initialize_models()


class NLPRequest(BaseModel):
    """Using to standardize the NLP Request

    :param BaseModel: Pydantic BaseModel
    :type BaseModel: pydantic.BaseModel
    """

    question: str
    memory: str


class MultimodalRequest(BaseModel):
    """Using to standardize the Multimodel Request

    :param BaseModel: Pydantic BaseModel
    :type BaseModel: pydantic.BaseModel
    """

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
    """QA Tool for Aleph Alpha

    :param request: NLPRequest
    :type request: NLPRequest
    :return: Response from Aleph Alpha
    :rtype: dict
    """
    logger.info("Starting NLP Request")
    # sent request to aleph alpha
    model = AlephAlphaModel(
        AlephAlphaClient(host="https://api.aleph-alpha.com", token=config["token"]),
        # You need to choose a model with qa support for this example.
        model_name="luminous-extended",
    )

    # important to remove all of the " and '. Otherwise the request will fail
    question = request.question.replace('"', "").replace("'", "")
    memory = request.memory.replace('"', "").replace("'", "")

    document = Document.from_text(memory)

    request = QaRequest(
        query=question,
        documents=[document],
    )

    result = model.qa(request)

    return result


# speech to text service
@app.post("/speech")
def speech(file: UploadFile):
    """Speech to Text Service

    :param file: Audio file
    :type file: Upload
    """

    logger.info("Starting Speech to Text Request")
    # save the file
    filename = f"tmp_raw/{uuid.uuid4()}.wav"
    with open(filename, "wb") as buffer:
        buffer.write(file.file.read())

    # convert the file to text
    text = speech_to_text(filename)

    return text


# get dict for a name.
@app.get("/dict/{name}")
def get_dict(name: str):
    """Get the dict for a name

    :param name: Name of the dict
    :type name: str
    :return: Dict for the name
    :rtype: dict
    """
    logger.info("Starting Dict Request")
    # load csv file and return as json.
    df = pd.read_csv(f"tmp_dict/{name}.csv")
    return df.to_json(orient="records")


@app.post("/multimodal")
# async def multimodal(request: MultimodalRequest):
async def multimodal(file: UploadFile):
    """Api endpoint for multimodal requests, which will be parsed and redirected to the aleph alpha API.

    :param request: Request containing the image and the question.
    :type request: MultimodalRequest
    :return: Response from the aleph alpha API.
    :rtype: str
    """
    logger.info("Starting Multimodal Request")
    logger.info("Saving file locally.")
    id = str(uuid.uuid4())
    # save file locally.
    file_path = f"tmp_img/{id}.{file.filename.split('.')[-1]}"
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    model = AlephAlphaModel(
        AlephAlphaClient(host="https://api.aleph-alpha.com", token=config["token"]),
        # You need to choose a model with qa support for this example.
        model_name="luminous-extended",
    )
    # img = str.encode(request.img)
    # img = ImagePrompt.from_bytes(img)
    img = ImagePrompt.from_file(file_path)
    prompt = [img]
    document = Document.from_prompt(prompt)
    logger.info("Convertion sucessful.")
    # request = QaRequest(query=request.question, documents=[document])
    request = QaRequest(query="Q: What is in the Picture? A:", documents=[document])
    logger.info("Sending Request to Aleph Alpha.")
    result = model.qa(request)
    logger.info("Request sucessful.", result)

    print(result)
    return result


# detect image.
@app.post("/detect_image")
async def detect_image(file: UploadFile):
    # get the image and save it locally.
    logger.info("Saving file locally.")
    id = str(uuid.uuid4())
    file_path = f"tmp_raw/{id}.{file.filename.split('.')[-1]}"
    save_path = f"tmp_processed/{id}.{file.filename.split('.')[-1]}"
    dict_path = f"tmp_dict/{id}.json"

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # call the service to detect the video.
    try:
        detect_single_image(file_path, save_path, dict_path)
        logger.info("Video processed.")

    except Exception as e:
        logger.error("Error processing video.", e)
        return {"error": "Error processing video."}

    # return the image id.
    return id


# object detection request.
@app.post("/detection_video")
async def detection_video(file: UploadFile):
    """An Uploaded Video file will be processed via hugging face detection and the results will be stored in the file system.

    :param file: Uploaded Video
    :type file: UploadFile
    :return: UUID of the processed file
    :rtype: str
    """
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
@app.get("/detection/{id}")
async def get_detected(id: str):
    """Get the detected video for a specific id.

    :param id: ID of the Video in uuid.
    :type id: str
    :return: Video with detected objects, marked via bounding box.
    :rtype: FileResponse
    """
    file = glob.glob(f"tmp_processed/{id}.*")[0]

    return FileResponse(file)


# get detected classes.
@app.get("/detection_classes/{id}")
async def get_detected_classes(id: str):
    """Get the detected Classes from the stored json file. Identification via UUID.

    :param id: uuid
    :type id: str
    :return: detected classes
    :rtype: dict
    """
    with open(f"tmp_dict/{id}.json", "r") as f:
        data = f.read()

    data = json.loads(data)
    df = pd.DataFrame(columns=["index", "frame", "detection_class", "detection_score"])
    y = 0
    for d in data:
        # save number
        number = d
        detections = data[number]["detection_class"]
        prob = data[d]["prob"]
        for x in detections:
            for p in prob:

                df.loc[y] = [y, number, x, p]
                y += 1

    # change to json.
    return df.to_json(orient="records")


# get frame with the most detected objects.
@app.get("/detection_frame/{id}")
async def get_detected_frame(id: str):
    """Get the frame with the most detected objects.

    :param id: uuid
    :type id: str
    :return: image with the most detected objects.
    :rtype: FileResponse
    """
    with open(f"tmp_dict/{id}.json", "r") as f:
        file = f.read()

    file = json.loads(file)
    lenght = 0
    frame = 0

    for key in file:
        lx = len(file[key]["detection_class"])
        if lx > lenght:
            lenght = lx
            # get the key
            frame = key

    return frame
