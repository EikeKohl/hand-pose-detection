import cv2
import mediapipe as mp
import os
import uuid
import pandas as pd
import argparse
from utils.angle import create_p1_p2_p3, find_angle
from utils.misc import create_folder, read_yaml_config

"""
This script can be used to perform live annotation of image data with hand landmarks for model training.
"""

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

parser = argparse.ArgumentParser(description="Image annotation tool arguments")
parser.add_argument(
    "--output_folder",
    required=True,
    help="The folder where to save 'annotations.csv' and the corresponding images",
    type=str,
)

parser.add_argument(
    "--save_images",
    required=False,
    help="Whether the annotated images should be saved or not. Default is False",
    type=bool,
    default=False,
)


def annotate_data(output_folder, save_images):
    """
    This function is used to annotate images with the mediapipe hand landmark detection model.
    The annotations will be saved to a csv file and can be used for further model training.

    Parameters
    ----------
    output_folder: The folder where to save 'annotations.csv' and the corresponding images (str)
    save_images: Whether the annotated images should be saved or not. Default is False (bool)

    Returns
    -------
    None

    """

    # Read labels.yml and map keys to labels in a dictionary
    key_label_map = {
        value: key for key, value in read_yaml_config("labels.yml").items()
    }

    # Create the headers for the annotations output table
    headers = (
        ["label"]
        + [f"{j}_{coord}" for j in range(21) for coord in ["x", "y", "z"]]
        + [f"angle_{i}" for i in range(1, 5)]
    )

    # Instantiate output dictionary
    annotations_output = {key: [] for key in headers}

    # Create output_folder recursively if it does not exist yet
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set path to annotations.csv
    annotations_csv = os.path.join(output_folder, "annotations.csv")

    # Convert keys to ints so they match the output of cv2.waitkeys()
    keys = [ord(key) for key in list(key_label_map.keys())]

    # The counter is used to access annotations_output within a for loop
    counter = 0

    # Start video captioning with cv2
    cap = cv2.VideoCapture(0)

    # Start the annotation using the mediapipe hand pose estimation model
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
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

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            annotated_image = image.copy()

            # Get the int value for a key if pressed
            key = cv2.waitKey(30) & 0xFF

            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                if key in keys:

                    # Write label to output
                    label = key_label_map[chr(key)]
                    annotations_output["label"].append(label)

                    # Write all landmark coordinates to output
                    for i, landmark in enumerate(
                        results.multi_hand_landmarks[0].landmark
                    ):
                        annotations_output[f"{i}_x"].append(landmark.x)
                        annotations_output[f"{i}_y"].append(landmark.y)
                        annotations_output[f"{i}_z"].append(landmark.z)

                    # calculate angle 1 (thumb & index)
                    p1, p2, p3 = create_p1_p2_p3(4, 0, 8, annotations_output, counter)
                    annotations_output["angle_1"].append(find_angle(p1, p2, p3) / 360.0)

                    # calculate angle 2 (index & middle)
                    p1, p2, p3 = create_p1_p2_p3(8, 5, 12, annotations_output, counter)
                    annotations_output["angle_2"].append(find_angle(p1, p2, p3) / 360.0)

                    # calculate angle 3 (middle & ring)
                    p1, p2, p3 = create_p1_p2_p3(12, 9, 16, annotations_output, counter)
                    annotations_output["angle_3"].append(find_angle(p1, p2, p3) / 360.0)

                    # calculate angle 4 (ring & pinky)
                    p1, p2, p3 = create_p1_p2_p3(
                        16, 13, 20, annotations_output, counter
                    )
                    annotations_output["angle_4"].append(find_angle(p1, p2, p3) / 360.0)

                    counter += 1

                    # Save the images if needed
                    if save_images:
                        img_folder = os.path.join(output_folder, label)
                        create_folder(img_folder)
                        cv2.imwrite(
                            img_folder + f"/{label}_" + uuid.uuid4().hex + ".png",
                            cv2.flip(image, 1),
                        )

                    print(f"image annotated: {label}")

            # Draw the hand annotations on the image.
            cv2.imshow("MediaPipe Hands", annotated_image)

            # Quit application if 'esc' is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

    # Write the annotations to 'annotations.csv'
    with open(annotations_csv, "a", newline="") as f:

        annotations_df = pd.DataFrame.from_dict(annotations_output)
        annotations_df.to_csv(f, index=False, mode="a", header=not f.tell())
        print(
            "The following training data has been generated: \n",
            annotations_df["label"].value_counts(),
        )


if __name__ == "__main__":

    args = parser.parse_args()
    annotate_data(args.output_folder, args.save_images)
