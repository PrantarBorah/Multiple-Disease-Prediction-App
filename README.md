# Multiple Disease Prediction App

This repository contains a small [Streamlit](https://streamlit.io) web application that predicts the likelihood of
three different diseases using pretrained machine learning models.
The app allows a user to enter clinical parameters and returns a simple diagnosis
for each of the following problems:

- **Diabetes** – predicts whether a patient is diabetic.
- **Heart Disease** – identifies if a patient suffers from heart disease.
- **Parkinson's Disease** – determines the presence of Parkinson's from voice measurements.

The goal of the project is to demonstrate how relatively light‑weight models can be
served with a user‑friendly interface.

## Repository Layout

| File | Description |
|------|-------------|
| `Diabetes_Prediction_Model_for_Streamlit.ipynb` | Notebook used to train an SVM model on a diabetes dataset. |
| `Heart_Disease_Prediction_for_Streamlit.ipynb` | Notebook that trains a logistic regression classifier for heart disease. |
| `Parkinsons_Disease_Prediction_for_Streamlit_pynb.ipynb` | Notebook training an SVM model for Parkinson's detection. |
| `Multi_DiseasePrediction_Streamlit.py` | Streamlit application loading the above models. |

Pretrained model files (`*.sav`) are expected to live next to the
application script. They are loaded using `pickle` when the app starts.

## Running the App

1. Install the dependencies listed in the notebooks (e.g. `streamlit`,
   `scikit-learn`, `pandas`).
2. Ensure the trained model files are present in the repository directory:
   `trained_model.sav`, `heart_disease_model.sav` and `parkinsons_model.sav`.
3. Launch the Streamlit app:

```bash
streamlit run Multi_DiseasePrediction_Streamlit.py
```

A sidebar will let you choose between diabetes, heart disease or
Parkinson's prediction. Input the requested clinical values and the model
outputs the corresponding diagnosis.

## Are the Models Sufficient?

The current models were created in the notebooks as quick examples using
small, public datasets. The diabetes and Parkinson's detectors are
**linear support vector machines**, while the heart disease classifier
uses **logistic regression**. These algorithms are simple and
interpretable, but their accuracy is limited by the size and quality of
the datasets. On basic train/test splits the notebooks report accuracies
around the mid to high 80% range.

For a production‑ready medical tool you would typically want more
extensive evaluation, cross‑validation and possibly more complex
algorithms (e.g. ensembles or neural networks) trained on larger and more
representative data. Therefore, while the current models are sufficient
for demonstrating the workflow, more sophisticated approaches may be
required for real‑world deployment.

## License

This project is shared for educational purposes. Please consult the
original dataset sources for their respective licenses.

