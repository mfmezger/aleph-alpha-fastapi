import copy
import json
import logging
from logging.config import dictConfig

import cv2
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection

from config import LogConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger("client")


def draw_on_image(results, img, model, score_confidence=0.9, debugging=False):
    color = [100, 128, 0]
    # save detection and time stamp
    detection_class = []
    prob = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # let's only keep detections with score > 0.9
        if score > score_confidence:
            if debugging:
                print(f"Detected {model.config.id2label[label.item()]} with confidence " f"{round(score.item(), 3)} at location {box}")
            # draw bouding box on img.
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                img,
                model.config.id2label[label.item()],
                (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_TRIPLEX,
                1,
                color,
                2,
                lineType=cv2.LINE_4,
            )
            detection_class.append(model.config.id2label[label.item()])
            prob.append(round(score.item(), 3))

    return img, detection_class, prob


def get_classes_in_image(path_to_image):
    image = Image.open(path_to_image)
    image = image.convert("RGB")

    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    img = cv2.imread(path_to_image)

    # draw on image.
    img = draw_on_image(results, img, model)

    return img


def detect_video(path_to_video, save_path, dict_path):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path_to_video)

    # initialize the model.
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    # feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    # model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')

    # initialize video with detection bounding boxes.
    codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(save_path, codec, fps, (width, height))

    detections = {}
    logger.info("Starting Predicitons.")
    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
    frame = 0
    results_dict = {}
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, image = cap.read()
        if ret == True:
            # process the video
            img = copy.deepcopy(image)
            image = Image.fromarray(image)

            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            # detaching all of the outputs
            target_sizes = torch.tensor([image.size[::-1]])
            results = feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]
            outputs = {k: v.detach().cpu() for k, v in outputs.items()}
            img, detection_class, prob = draw_on_image(results, img, model, debugging=False)
            output_video.write(img)
            # store prob and detection_class with frame in dict.
            detections[frame] = {"detection_class": detection_class, "prob": prob}

            # convert outputs (bounding boxes and class logits) to COCO API
            frame += 1
            results_dict[frame] = results

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    # # draw results.
    # cap = cv2.VideoCapture(path_to_video)
    # while cap.isOpened():
    #     # Capture frame-by-frame
    #     ret, image = cap.read()
    #     if ret == True:
    #         results = results_dict[frame]

    #         img, detection_class, prob = draw_on_image(results, image, model, debugging=False)
    #         output_video.write(img)
    #         # store prob and detection_class with frame in dict.
    #         detections[frame] = {"detection_class": detection_class, "prob": prob}

    output_video.release()
    logger.info("Writing Results to File.")

    # convert dict to json.
    with open(dict_path, "w") as f:
        json.dump(detections, f)

    # Closes all the frames
    cv2.destroyAllWindows()


def main():
    # path_to_image = "cat.jpg"

    # image = Image.open(path_to_image)
    # image = image.convert("RGB")

    # img = cv2.imread(path_to_image)
    # # draw on image.
    # img = get_classes_in_image(path_to_image)

    # cv2.imwrite("cat_detected.jpg", img)

    # save image
    path_to_video = "very_short.mp4"
    save_path = "very_short_detected.mp4"
    dict_path = "tmp_dict/tst.json"
    detect_video(path_to_video, save_path, dict_path)


if __name__ == "__main__":
    main()
