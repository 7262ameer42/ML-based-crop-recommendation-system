# 🌱 Crop Recommendation System

A machine learning project that recommends the best crop to grow based on **soil nutrients** (Nitrogen, Phosphorus, Potassium, pH) and **weather conditions** (temperature, humidity, rainfall).

## Project Description

Choosing the right crop for a given soil and climate condition is a critical decision for farmers. This project uses a **Random Forest Classifier** trained on soil and weather data to recommend the most suitable crop out of 12 common crops (rice, wheat, maize, cotton, sugarcane, chickpea, potato, tomato, mango, banana, coffee, jute).

The project includes:
- A Jupyter/Colab notebook covering the full ML workflow (data → training → evaluation → prediction)
- A trained model saved as a `.pkl` file, ready for reuse
- An optional Streamlit web app for interactive predictions

## Features

- Predicts the best crop from soil + weather inputs
- Shows top-3 recommended crops with confidence scores
- Visualizes feature correlations, confusion matrix, and feature importance
- Interactive Streamlit UI for non-technical users
- Easily retrainable on your own dataset (e.g. Kaggle's Crop Recommendation Dataset)

## Dataset

`data/Crop_recommendation.csv` — synthetically generated using realistic agronomic ranges for each crop's ideal N, P, K, pH, temperature, humidity, and rainfall values (2,400 rows, 12 crops × 200 samples). You can swap this out for a real dataset (e.g. Kaggle) with the same column structure: `N, P, K, temperature, humidity, ph, rainfall, label`.

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit (web app)
- Jupyter Notebook / Google Colab

## Machine Learning Algorithm

**Random Forest Classifier** — an ensemble of decision trees, chosen for its strong performance on tabular data with mixed feature scales and its built-in feature importance scores.

- Train/test split: 80/20 (stratified)
- Achieved **~91% accuracy** on the held-out test set

## Installation

```bash
git clone https://github.com/<your-username>/Crop-Recommendation-System.git
cd Crop-Recommendation-System
pip install -r requirements.txt
```

## Run

**Notebook:**
```bash
jupyter notebook notebook/Crop_Recommendation_ML.ipynb
```
Or upload it directly to [Google Colab](https://colab.research.google.com).

**Web app:**
```bash
streamlit run app.py
```

## Results

- **Accuracy:** ~91% on test data
- Confusion matrix and feature importance charts available in `images/`

| Confusion Matrix | Feature Importance |
|---|---|
| ![Confusion Matrix](images/confusion_matrix.png) | ![Feature Importance](images/feature_importance.png) |

## Future Improvements

- Train on a larger, real-world agricultural dataset
- Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Add more crops and region-specific soil/weather ranges
- Deploy the Streamlit app publicly (Streamlit Community Cloud / Render)
- Add unit tests and CI workflow

## License

This project is licensed under the [MIT License](LICENSE).
