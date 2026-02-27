# GastroCAD - AI Based Stomach Cancer Detection

## Overview
GastroCAD is a Flask-based AI system for gastrointestinal disease classification using DenseNet201.

## Features
- User Registration & Login (SQLite)
- DenseNet201 Deep Learning Model
- REST API for Prediction
- GPU/CPU Compatible Deployment

## Model Details
- Architecture: DenseNet201
- Classes: 8 GI disease categories
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Regularization: Dropout(0.5)

## Run Locally

pip install -r requirements.txt
python app.py

## API Endpoints

POST /register  
POST /login  
POST /predict (form-data key: image)

## Model File
models/Stomachcancer.pth
