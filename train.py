import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import seaborn as sn
from utils.misc import read_multiple_csv, read_yaml_config

DATA_DIRECTORY = "./data4"


csv_data = [
    os.path.join(DATA_DIRECTORY, file)
    for file in os.listdir(DATA_DIRECTORY)
    if file.split(".")[-1] == "csv"
]
data = np.array(read_multiple_csv(csv_data))


X_train, X_test, y_train, y_test = train_test_split(
    data[:, 1:], data[:, 0], stratify=data[:, 0], test_size=0.2, random_state=42
)

config = read_yaml_config("labels.yml")
for idx, key in enumerate(config.keys()):

    y_train = np.where(y_train == key, idx, y_train)
    y_test = np.where(y_test == key, idx, y_test)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

tf.keras.backend.clear_session()

model = tf.keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(67,)),
        layers.Dropout(0.6),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(config.keys()), activation="softmax"),
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=60,
    verbose=1,
    validation_data=(X_test, y_test),
)


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.show()

predictions = np.argmax(model.predict(X_test), axis=1)


cm = tf.math.confusion_matrix(
    y_test, predictions, num_classes=8, weights=None, dtype=tf.dtypes.int32, name=None
)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.show()

model.save("model")
