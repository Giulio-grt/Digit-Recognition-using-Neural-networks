<img width="500" alt="logo" src="assets/cover.png">

# Digit Recognition using neural networks

Welcome to this brief educational project ! It demonstrates how a convolutional neural network, built with TensorFlow, can recognize handwritten digits.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=purple)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?&style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)


## This repository will walk you through
- Data preprocessing before feeding it into a machine learning model
- The basics of building a simple convolutional neural network 
- Building a simple web app to try-out the model yourself ! 

This project assumes only a very basic understanding of Python (mainly data handling with Pandas and NumPy) and is inspired by the tensorflow short videos: ML zero to hero (part 1 to 4) which I highly recommend you check out !

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtu.be/KNAWp2S3w94?feature=shared)

## Project structure

```
├── README.md
├── assets
│   └── cover.png
├── src
│   ├── frontend                        
│   │   └── main.py                     #Code for the streamlit frontend
│   └── model               
│       ├── data                        #All the data needed to train and test the model
│       ├── train.ipynb                 #Notebook for training the model
│       └── trained_model.h5            #Trained model to perform the predictions
```

# Project architecture

```mermaid
flowchart TB
    subgraph Repo
        R[README.md]
        subgraph Assets
            A[cover.png]
        end
        subgraph Src
            subgraph Frontend
                F[main.py<br/>Streamlit UI]
            end
            subgraph Model
                D[data/]
                T[train.ipynb<br/>Training Notebook]
                M[trained_model.h5<br/>Saved Model]
            end
        end
    end

    %% Flow
    D --> T --> M --> F
```


