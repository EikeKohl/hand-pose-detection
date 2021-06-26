import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from utils.angle import create_p1_p2_p3, find_angle
from utils.misc import read_yaml_config

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model = tf.keras.models.load_model("model")

config = read_yaml_config("labels.yml")
label_map = {idx: key for idx, key in enumerate(config.keys())}

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
        annotations_output = {}
        ###########################################################

        if results.multi_hand_landmarks:

            for i, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
                annotations_output[f"{i}_x"] = landmark.x
                annotations_output[f"{i}_y"] = landmark.y
                annotations_output[f"{i}_z"] = landmark.z

            # calculate angle 1
            p1, p2, p3 = create_p1_p2_p3(4, 0, 8, annotations_output)
            annotations_output["angle_1"] = find_angle(p1, p2, p3) / 360.0

            # calculate angle 2
            p1, p2, p3 = create_p1_p2_p3(8, 5, 12, annotations_output)
            annotations_output["angle_2"] = find_angle(p1, p2, p3) / 360.0

            # calculate angle 3
            p1, p2, p3 = create_p1_p2_p3(12, 9, 16, annotations_output)
            annotations_output["angle_3"] = find_angle(p1, p2, p3) / 360.0

            # calculate angle 4
            p1, p2, p3 = create_p1_p2_p3(16, 13, 20, annotations_output)
            annotations_output["angle_4"] = find_angle(p1, p2, p3) / 360.0

            predictor_input = np.array(list(annotations_output.values())).reshape(
                (1, 67)
            )
            raw_prediction = model.predict(predictor_input)
            prediction = np.argmax(raw_prediction)

            cv2.putText(
                image,
                label_map[prediction],
                (70, 50),
                cv2.FONT_ITALIC,
                3,
                (255, 0, 0),
                3,
            )

        ###########################################################

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
