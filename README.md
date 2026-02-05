🍽️ Restaurant Rating Clustering Backend

A backend service for clustering restaurant ratings using machine learning. This project trains a K-Means clustering model on restaurant rating data and exposes an API to get clustered predictions for new restaurant inputs.

📌 Project Overview

This repository contains:

A dataset of restaurant ratings (Dataset .csv)

A trained K-Means model (kmeans_model.pkl)

A saved scaler (scaler.pkl)

Python script for training the model

API code to serve clustering predictions

Configuration for deployment

📂 Repository Structure
Restaurant_Rating_Clustering_Backend/
├── api/                      # FastAPI or Flask backend code
├── Dataset .csv              # Restaurant rating dataset
├── kmeans_model.pkl          # Pre-trained clustering model
├── scaler.pkl                # Pre-trained scaler for feature scaling
├── model_training.py         # Script to train model
├── requirements.txt          # Python dependencies
└── render.yaml               # Deployment configuration

🚀 Features

✔ Trains a K-Means clustering model
✔ Saves scaler & model for inference
✔ Provides backend API to classify restaurants into clusters
✔ Easy deployment configuration

🧠 How It Works

Load Dataset
The dataset contains restaurant attributes such as ratings, features, etc.

Preprocess Data
Scale and clean data using scaler.pkl.

Train Model
Use model_training.py to build and serialize the K-Means model.

Serve Predictions
The API (in the api/ folder) loads the saved model and scaler to return cluster labels for new data.

📦 Installation

Clone the repository:

git clone https://github.com/princekr99/Restaurant_Rating_Clustering_Backend.git
cd Restaurant_Rating_Clustering_Backend


Create a virtual environment:

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

📌 Usage
🛠 Train the Model
python model_training.py


This will:

Load the dataset

Scale features

Train a K-Means model

Save the model and scaler

🚀 Run the API

Depending on your backend (FastAPI / Flask), start the server:

Example (FastAPI):

uvicorn api.main:app --reload


Once running, you can request cluster predictions.

📊 API Example

Request

POST /predict
{
  "feature1": value,
  "feature2": value,
  ...
}


Response

{
  "cluster_label": 2
}

🛠 Deployment

A basic deployment config is provided in render.yaml. You can deploy this app to Render
 or similar cloud platforms.

🧩 Dependencies

Dependencies are listed in requirements.txt, typically including:

scikit-learn

pandas

numpy

Flask or FastAPI

joblib (for model saving)

uvicorn (if FastAPI is used)
 
📈 Model Insights

This project focuses on unsupervised clustering using restaurant rating data. Restaurants with similar rating patterns and features will be grouped into meaningful clusters using K-Means.

📄 License

This project is open-source — feel free to use, modify, and distribute. Add a license file as needed.

🙌 Contributions

Contributions and improvements are welcome! Feel free to submit pull requests or open issues.
