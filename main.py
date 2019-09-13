import numpy as np
import time
import cv2
from imutils.video import VideoStream
import imutils
import nxt

import logging
import chromalog
from chromalog.mark.helpers.simple import success

chromalog.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

logger.debug("connecting to NXT brick...")
brick = nxt.locator.find_one_brick(debug=True)

# motors for rotating
logger.debug("setting up motors")
vertical = nxt.motor.Motor(brick, nxt.PORT_A)
horizontal = nxt.motor.Motor(brick, nxt.PORT_B)

logger.debug("loading model")
# model used was made by generous folks over at pyimagesearch.com. thanks so much!
net = cv2.dnn.readNetFromCaffe("./prototext.txt", "./model.caffemodel")
logger.debug("model loaded!")

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
logger.debug("connecting to webcam...")
vs = VideoStream(src=0).start()
time.sleep(2)

# pass the blob through the network and obtain the detections and
# predictions
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            PERSON_INDEX = 15  # other indexes are other objects (bottles etc)
            if idx == PERSON_INDEX:
                logger.debug("found a person in range. calculating offset...")
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                middleX = endX - (endX - startX) / 2
                middleY = (
                    # endY - (endY - startY)
                    startY
                    + (endY - startY) / 4
                )  # we want to focus face, not stomach

                offsetX = w / 2 - middleX
                offsetY = h / 2 - middleY

                logger.debug("offset: X: %s Y: %s", success(offsetX), success(offsetY))
                if abs(offsetY) < 5 and abs(offsetX) < 5:
                    logger.debug("position is perfect!")
                else:
                    # rotate here
                    logger.debug("adjusting position...")
                    speed = 40
                    try:
                        if offsetY > 0:
                            vertical.turn(-speed, abs(offsetY))
                        else:
                            vertical.turn(speed, abs(offsetY))

                        if offsetX > 0:
                            horizontal.turn(-speed, abs(offsetX))
                        else:
                            horizontal.turn(speed, abs(offsetX))
                    except:
                        logger.error("there was an error rotatng motors!")
                        logger.error("most likely they got blocked")


cv2.destroyAllWindows()
vs.stop()
