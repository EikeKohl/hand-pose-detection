import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import seaborn as sn
from utils.misc import read_multiple_csv, read_yaml_config


"""
This script is used to train a small Fully Connected DNN using the annotated training data from annotate.py.
"""

parser = argparse.ArgumentParser(description="Model training arguments")
parser.add_argument(
    "--data_directory",
    required=True,
    help="The directory where 'annotations.csv' is saved",
    type=str,
)
parser.add_argument(
    "--show_plots",
    required=False,
    help="Whether to show accuracy and loss plots as well as the confusion matrix or not after training",
    type=bool,
    default=False,
)


def train_model(data_directory, show_plots):

    """
    This function trains a model for hand pose classification using 21 hand landmarks and calculated angles between
    all fingers.

    Parameters
    ----------
    data_directory: The directory where 'annotations.csv' is saved (str)
    show_plots: Whether to show accuracy and loss plots as well as the confusion matrix or not after training (bool)

    Returns
    -------
    The model training history
    """

    # Define list of all csv files within data_directory in case multiple annotations files exist.
    csv_data = [
        os.path.join(data_directory, file)
        for file in os.listdir(data_directory)
        if file.split(".")[-1] == "csv"
    ]

    # Read all csv files
    data = np.array(read_multiple_csv(csv_data))

    # Create train / test split
    x_train, x_test, y_train, y_test = train_test_split(
        data[:, 1:], data[:, 0], stratify=data[:, 0], test_size=0.2, random_state=42
    )

    # Preprocess training and test data
    config = read_yaml_config("labels.yml")
    for idx, key in enumerate(config.keys()):
        y_train = np.where(y_train == key, idx, y_train)
        y_test = np.where(y_test == key, idx, y_test)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Clear keras sessions to ensure a clean model training
    tf.keras.backend.clear_session()

    # Setup and compile small fully connected sequential tensorflow model
    num_classes = len(config.keys())
    model = tf.keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(67,)),
            layers.Dropout(0.6),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model.summary()

    # Fit model
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=60,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    if show_plots:

        # Accuracy plots
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.legend(["accuracy", "val_accuracy"])
        plt.show()

        # Loss plots
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.legend(["loss", "val_loss"])
        plt.show()

        # Confusion Matrix
        predictions = np.argmax(model.predict(x_test), axis=1)
        cm = tf.math.confusion_matrix(
            y_test,
            predictions,
            num_classes=num_classes,
            weights=None,
            dtype=tf.dtypes.int32,
            name=None,
        )
        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True)
        plt.show()

    # Save model in folder 'model'
    model.save("model")

    return history


if __name__ == "__main__":

    args = parser.parse_args()
    history = train_model(args.data_directory, args.show_plots)
