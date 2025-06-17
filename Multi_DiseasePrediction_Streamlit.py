import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Import prediction functions from new model files
from parkinsons_model import parkinsons_prediction
from diabetes_model import diabetes_prediction
from heart_disease_model import heart_disease_prediction

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        # Ensure all inputs are converted to float
        try:
            user_input_float = [float(x) for x in user_input]
            diab_diagnosis = diabetes_prediction(user_input_float)
        except ValueError:
            diab_diagnosis = "Please enter valid numerical values for all fields."

    st.success(diab_diagnosis)

    st.subheader("Feature Importances (Diabetes Model)")
    try:
        diabetes_feature_importances = pickle.load(open('diabetes_feature_importances.pkl', 'rb'))
        diabetes_feature_names = pickle.load(open('diabetes_feature_names.pkl', 'rb'))

        # Create a DataFrame for better visualization
        feature_df = pd.DataFrame({
            'Feature': diabetes_feature_names,
            'Importance': diabetes_feature_importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_df['Feature'], feature_df['Importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Diabetes Model Feature Importances')
        plt.gca().invert_yaxis() # Display most important at the top
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Run diabetes_model.py first to generate feature importance data.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        # Ensure all inputs are converted to float
        try:
            user_input_float = [float(x) for x in user_input]
            heart_diagnosis = heart_disease_prediction(user_input_float)
        except ValueError:
            heart_diagnosis = "Please enter valid numerical values for all fields."

    st.success(heart_diagnosis)

    st.subheader("Feature Importances (Heart Disease Model)")
    try:
        heart_feature_importances = pickle.load(open('heart_disease_feature_importances.pkl', 'rb'))
        heart_feature_names = pickle.load(open('heart_disease_feature_names.pkl', 'rb'))

        feature_df = pd.DataFrame({
            'Feature': heart_feature_names,
            'Importance': heart_feature_importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_df['Feature'], feature_df['Importance'], color='lightcoral')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Heart Disease Model Feature Importances')
        plt.gca().invert_yaxis()
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Run heart_disease_model.py first to generate feature importance data.")

# Parkinson's Prediction page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinsons Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')
    

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                    RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                     APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        
        # Ensure all inputs are converted to float
        try:
            user_input_float = [float(x) for x in user_input]
            parkinsons_diagnosis = parkinsons_prediction(user_input_float)
        except ValueError:
            parkinsons_diagnosis = "Please enter valid numerical values for all fields."

    st.success(parkinsons_diagnosis)

    st.subheader("Feature Importances (Parkinsons Model)")
    try:
        parkinsons_feature_importances = pickle.load(open('parkinsons_feature_importances.pkl', 'rb'))
        parkinsons_feature_names = pickle.load(open('parkinsons_feature_names.pkl', 'rb'))

        feature_df = pd.DataFrame({
            'Feature': parkinsons_feature_names,
            'Importance': parkinsons_feature_importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feature_df['Feature'], feature_df['Importance'], color='lightgreen')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Parkinsons Model Feature Importances')
        plt.gca().invert_yaxis()
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Run parkinsons_model.py first to generate feature importance data.")
    
        

