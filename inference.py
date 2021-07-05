import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from utils.angle import create_p1_p2_p3, find_angle
from utils.misc import read_yaml_config

"""
This script is used to perform the actual hand pose detection. The detected hand poses could be used to control or start
a program.
"""

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def detect_hand_poses():
    """
    This function is used to detect hand poses and show them on screen. The detected poses could be used to control
    or start a program

    Returns
    -------
    None
    """

    # Load the trained model
    model = tf.keras.models.load_model("model")

    # Load config with commands
    config = read_yaml_config("labels.yml")
    label_map = {idx: key for idx, key in enumerate(config.keys())}

    # Start Video Capturing
    cap = cv2.VideoCapture(0)

    # Detect hand landmarks
    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Create empty dictionary for annotations to fill it with data for the model
            annotations_output = {}

            if results.multi_hand_landmarks:

                # Save hand landmark coordinates
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

                # Convert annotations to numpy array with required shape
                predictor_input = np.array(list(annotations_output.values())).reshape(
                    (1, 67)
                )

                # Predict handpose
                raw_prediction = model.predict(predictor_input)
                print(raw_prediction)
                prediction = np.argmax(raw_prediction)

                # Write the result to the live video
                cv2.putText(
                    image,
                    label_map[prediction],
                    (70, 50),
                    cv2.FONT_ITALIC,
                    3,
                    (255, 0, 0),
                    3,
                )

            # Draw the hand landmark annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
            cv2.imshow("MediaPipe Hands", image)

            # Press 'esc' to stop hand pose detection
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


if __name__ == "__main__":

    detect_hand_poses()
