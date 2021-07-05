# hand-pose-detection

This repository can be used to annotate data, train a model and use the resulting model for live video hand pose detection.

## Setup Python Environment

  First, install all the requirements:

```bash
conda create -n handposedetection python=3.6
conda activate handposedetection
pip install -r requirements.txt
```

## Annotate Your Data

  To annotate your data, please first set up the ```labels.yml```. The keys are the names of the commands / hand poses.
  As values, please assign keys you want to use for annotation.

  Once this is done, you can start the annotation via CLI as follows:

```bash
python annotate.py --output_folder "data" --save_images False
```

  The annotation script will save 21 coordinates for hand landmarks plus 4 calculated angles and the respective label
  to ```<output_folder>/annotations.csv```. You can also save the corresponding images of your training data by setting 
  ```--save_images``` to ```True```.

## Train The Model

  The model trained to perform hand pose detection is a shallow fully connected model with 3 hidden layers. It will
  output a hand pose prediction as softmax output. The resulting model will be saved in a folder called ```model```.
  To start the model training, please use the following command:

```bash
python train.py --data_directory "data" --show_plots False
```

  The model will use all csv files with annotations in ```data_directory```. If you want to monitor the model training,
  you can set ```show_plots``` to ```True```.

## Detect Hand Poses

  Now you can use your trained model to detect hand poses live through your webcam. To do so, just start ```inference.py```:

```bash
python inference.py
```

  Have fun!

## To Do

* build docker image
* create executable
* Implement minimum probability for hand pose detection
