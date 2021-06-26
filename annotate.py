import cv2
import mediapipe as mp
import os
import uuid
import pandas as pd
from utils.angle import create_p1_p2_p3, find_angle
from utils.misc import create_folder

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

OUTPUT_FOLDER = "./data4"
KEY_LABEL_MAP = {
    "g": "greifen",
    "e": "ein",
    "a": "aus",
    "s": "schneller",
    "l": "langsamer",
    "r": "richtungsaenderung",
    "n": "nothalt",
    "x": "no_gesture",
}

headers = (
    ["label"]
    + [f"{j}_{coord}" for j in range(21) for coord in ["x", "y", "z"]]
    + [f"angle_{i}" for i in range(1, 5)]
)
annotations_output = {key: [] for key in headers}
annotations_csv = os.path.join(OUTPUT_FOLDER, "annotations.csv")
keys = [ord(key) for key in list(KEY_LABEL_MAP.keys())]
counter = 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

            # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        ###########################################################

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        annotated_image = image.copy()

        key = cv2.waitKey(30) & 0xFF

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            if key in keys:

                label = KEY_LABEL_MAP[chr(key)]
                annotations_output["label"].append(label)

                for i, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                    annotations_output[f"{i}_x"].append(landmark.x)
                    annotations_output[f"{i}_y"].append(landmark.y)
                    annotations_output[f"{i}_z"].append(landmark.z)

                # calculate angle 1
                p1, p2, p3 = create_p1_p2_p3(4, 0, 8, annotations_output, counter)
                annotations_output["angle_1"].append(find_angle(p1, p2, p3) / 360.0)

                # calculate angle 2
                p1, p2, p3 = create_p1_p2_p3(8, 5, 12, annotations_output, counter)
                annotations_output["angle_2"].append(find_angle(p1, p2, p3) / 360.0)

                # calculate angle 3
                p1, p2, p3 = create_p1_p2_p3(12, 9, 16, annotations_output, counter)
                annotations_output["angle_3"].append(find_angle(p1, p2, p3) / 360.0)

                # calculate angle 4
                p1, p2, p3 = create_p1_p2_p3(16, 13, 20, annotations_output, counter)
                annotations_output["angle_4"].append(find_angle(p1, p2, p3) / 360.0)

                counter += 1

                img_folder = os.path.join(OUTPUT_FOLDER, label)
                create_folder(img_folder)

                cv2.imwrite(
                    img_folder + f"/{label}_" + uuid.uuid4().hex + ".png",
                    cv2.flip(image, 1),
                )

                print("annotated image")

        ###########################################################

        # Draw the hand annotations on the image.

        cv2.imshow("MediaPipe Hands", annotated_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

annotations_df = pd.DataFrame.from_dict(annotations_output)
annotations_df.to_csv(annotations_csv, index=False, mode="a")
