import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression

MIN_CONFIDENCE = 0.5
WIDTH = 320
HEIGHT = 320
layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]


def deskew(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    cv2.putText(
        rotated,
        "Angle: {:.2f} degrees".format(angle),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    return rotated


def iou(box1, box2):

    xtl = max(box1[0], box2[0])
    ytl = max(box1[1], box2[1])
    xbr = min(box1[2], box2[2])
    ybr = min(box1[3], box2[3])

    height = xbr - xtl
    width = ybr - ytl
    intersection_area = max(height, 0) * max(width, 0)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area

    return intersection_area / union_area


def detect_unstructured_data(image):
    img = cv2.imread(image)
    resized_img = cv2.resize(img, (WIDTH, HEIGHT))

    width_ratio = img.shape[1] / WIDTH
    height_ratio = img.shape[0] / HEIGHT

    east_net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    img_blob = cv2.dnn.blobFromImage(
        resized_img,
        1.0,
        (WIDTH, HEIGHT),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )
    east_net.setInput(img_blob)
    scores, coordinate_angle = east_net.forward(layer_names)

    rows, cols = scores.shape[2], scores.shape[3]
    boxes = []
    Angles = []
    confidences = []

    for i in range(rows):
        score = scores[0, 0, i]
        yCoordinateT = coordinate_angle[0, 0, i]
        xCoordinateR = coordinate_angle[0, 1, i]
        yCoordinateB = coordinate_angle[0, 2, i]
        xCoordinateL = coordinate_angle[0, 3, i]
        angles = coordinate_angle[0, 4, i]

        for j in range(cols):
            if score[j] < MIN_CONFIDENCE:
                continue

            angle = angles[j]
            cos = np.cos(angle)
            sin = np.sin(angle)

            BrX = int(j * 4.0 + (xCoordinateR[j] * cos) + (yCoordinateB[j] * sin))
            BrY = int(i * 4.0 - (xCoordinateR[j] * sin) + (yCoordinateB[j] * cos))
            TlX = int(BrX - (xCoordinateL[j] + xCoordinateR[j]))
            TlY = int(BrY - (yCoordinateT[j] + yCoordinateB[j]))

            boxes.append((TlX, TlY, BrX, BrY))
            Angles.append(angle)
            confidences.append(score[j])

    boxes = non_max_suppression(np.array(boxes), confidences)
    # boxes = cv2.dnn.NMSBoxesRotated(np.array(boxes), confidences, 0.5, 0.5)

    for box in boxes:
        (TlX, TlY, BrX, BrY) = box
        TlX = int(TlX * width_ratio)
        TlY = int(TlY * height_ratio)
        BrX = int(BrX * width_ratio)
        BrY = int(BrY * height_ratio)

        roi = img[TlY:BrY, TlX:BrX]
        configuration = "-l eng --oem 1 --psm 8"
        text = pytesseract.image_to_string(roi, config=configuration)
        print(text)

        cv2.rectangle(img, (TlX, TlY), (BrX, BrY), (0, 255, 0), 2)
    return img
