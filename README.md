
# 🌱 AgriSmart AI — Crop Recommendation System

An interactive machine learning application that recommends suitable crops based on soil nutrients and weather conditions.

## Features

- Predicts suitable crops using a Random Forest classifier
- Accepts seven input parameters: Nitrogen, Phosphorus, Potassium, temperature, humidity, soil pH, and rainfall
- Displays the predicted crop and top three recommendations
- Shows model probability estimates
- Interactive dashboard built with Streamlit

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## Project Structure

```text
CropRecommendationSystem/
├── app.py
├── train_model.py
├── compare_models.py
├── requirements.txt
├── README.md
├── model/
│   ├── crop_model.pkl
│   └── features.pkl
└── .gitignore
```

## Run Locally

1. Clone this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install the dependencies.
5. Run the application.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Crop Recommendation Dataset from Kaggle:

https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Check the dataset's license and attribution requirements before redistributing it.

## Model Evaluation

The project includes scripts for training the Random Forest model and comparing classification algorithms.

Evaluation results depend on the dataset and train-test split. Model probabilities are estimates and do not guarantee real-world crop suitability.

## Future Improvements

- Add location-based weather information
- Improve the user interface
- Add crop suitability explanations
- Deploy the application online

## Author

Computer Engineering Student | Machine Learning Project
