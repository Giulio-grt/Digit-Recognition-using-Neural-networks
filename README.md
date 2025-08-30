<img width="500" alt="logo" src="assets/cover.png">

# Digit Recognition using neural networks

Welcome to this brief educational project on the basics of computer vision ! Here, we demonstrate how a convolutional neural network (CNN), built with TensorFlow, can recognize handwritten digits.

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


├── LICENSE
├── README.md
├── assets
│   └── cover.png
├── src
│    ├── frontend                         #Code for the streamlit frontend
│    │   └── main.py
│    └── model
│        ├── data                         #Data needed to train and test the model
│        │   ├── test_data.csv
│        │   └── train_data.csv
│        ├── train.ipynb                  #Notebook for training the model
│        └── trained_model.h5             #Trained model to perform the predictions
└── environment.yml                       #conda environment dependencies
```

## Project architecture

```mermaid
flowchart TB
    subgraph Src
        subgraph Model
            D1[train_data.csv]
            D2[test_data.csv]
            T[train.ipynb<br/>Training Notebook]
            M[trained_model.h5<br/>Saved Model]
        end
        subgraph Frontend
            F[main.py<br/>Streamlit UI]
        end
    end

    %% Workflow with labels
    D1 -- "train" --> T -- "produces" --> M
    M -- "evaluate" --> D2
    M -- "serve" --> F
```

## Getting started

### Prerequisites
- Python 3.10
- Conda with `conda-forge` channel enabled : 

[![Download Anaconda](https://img.shields.io/badge/Download-Anaconda-44A833?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/download)


### 1) Clone the repo
```
git clone https://github.com/Giulio-grt/Digit-Recognition-using-Neural-networks.git
```


### 2) Create the environment with all prerequisties
```
conda env create -f environment.yml
conda activate digit
```

### 3) Understand how the model was built
Open the training notebook and run the cells:

- File: `src/model/train.ipynb`

> **Note:** You don’t need to install Conda or Python to simply *read* the notebook.  
> GitHub renders Jupyter notebooks directly in the browser, so you can open `src/model/train.ipynb` and scroll through the code and explanations without setting up anything locally.   

### 4) Run the web app
Naviguate to the correct folder :
```
cd Digit-Recognition-using-Neural-networks
```
Open and run the web app:
```
streamlit run src/frontend/main.py
```

## A small introduction to machine learning 

Machine Learning (ML) is about teaching computers to recognize patterns from data, rather than programming them with strict rules.  

In our case, instead of telling the computer *how* to recognize a digit, we show it many **examples** of handwritten digits along with the correct labels, and let it **learn the patterns** that distinguish a "3" from an "8" or a "7" from a "1".

At a very high level, the process looks like this:

1. **Data** – A dataset of handwritten digits (images + labels).
2. **Preprocessing** – Convert the images into numerical arrays so a computer can work with them.
3. **Model** – A *Convolutional Neural Network (CNN)*, which is especially good at learning from images.
4. **Training** – The model adjusts its internal parameters to minimize errors on the training data.
5. **Evaluation** – The model is tested on unseen data (test set) to measure accuracy.
6. **Prediction** – Once trained, the model can classify new handwritten digits it has never seen before.


This project shows one of the simplest yet most classic ML applications — handwritten digit recognition.  
Once you understand this workflow, you’ll be ready to explore more advanced ML applications in computer vision, natural language processing, and beyond 🚀

## Interested in applying ML to more technical domains?

Check out **RetroChem**, a project exploring machine learning for retrosynthesis in chemistry 🧪👨🏻‍🔬:

[![GitHub](https://img.shields.io/badge/GitHub-RetroChem-181717?style=for-the-badge&logo=github)](https://github.com/Flo-fllt/RetroChem)
