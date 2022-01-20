# Reading from a visual content with Python Open-CV

## About this app

This app will perform training a model to predict an alphabet correctly using Open-CV. This project is for the intermediate Python developer & Data Science newbi's. This will use a much known CNN models to prepare the model. There will be a series of posts on Open-CV will be present in the future.


## How to run this app

(The following instructions apply to Posix/bash. Windows users should check
[here](https://docs.python.org/3/library/venv.html).)

First, clone this repository and open a terminal inside the root folder.

Create and activate a new virtual environment (recommended) by running
the following:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```
You need to have the following directory to run this application correctly 

```
.
├── Data
│   └── A_Z_Handwritten_Data.csv
├── Model
│   └── model_trained_8.p
├── README.md
├── clsAlphabetReading.py
├── clsConfig.py
├── clsL.py
├── log
│   └── restoreVideo.log
├── readingVisualData.py
├── requirements.txt
└── trainingVisualDataRead.py

```

Run the model training for Visual Reading-App:

```bash
python trainingVisualDataRead.py
```
Once the model generates successfully, you should use the following commands to read the Alphabet using your webcam.

```bash
python readingVisualData.py
```

Make sure that you are properly connected with a functional WebCam (Preferably a separate external WebCam).

## Screenshots

![Demo.GIF](Demo.GIF)

## Resources

- To learn more about Open-CV, check out our [documentation](https://opencv.org/opencv-free-course/).
- To learn more about Matplotlib, check out our [documentation](https://matplotlib.org/stable/contents.html).
- To learn more about Tensorflow, check out our [documentation](https://www.tensorflow.org/tutorials).
- To learn more about Keras, check out our [documentation](https://keras.io/guides/).
- To learn more about Pickle, check out our [documentation](https://docs.python.org/3/library/pickle.html).
- To learn more about Scikit-Learn, check out our [documentation](https://scikit-learn.org/stable/tutorial/index.html).
