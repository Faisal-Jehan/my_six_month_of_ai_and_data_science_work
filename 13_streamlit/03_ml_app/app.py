import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import numpy as np
import io

# Function to load example data
def load_example_data(dataset_name):
    if dataset_name == 'titanic':
        return sns.load_dataset('titanic')
    elif dataset_name == 'tips':
        return sns.load_dataset('tips')
    elif dataset_name == 'iris':
        return sns.load_dataset('iris')

# Function to print basic data information
def print_data_info(data):
    st.write("### Data Head")
    st.write(data.head())
    st.write("### Data Shape")
    st.write(data.shape)
    st.write("### Data Description")
    st.write(data.describe())
    st.write("### Data Info")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("### Column Names")
    st.write(data.columns.tolist())

# Application starts here
st.title("Welcome to the Machine Learning Application")
st.write("""
This application allows you to perform machine learning tasks using your own dataset or an example dataset. 
You can preprocess the data, train models, and evaluate their performance.
""")

data_source = st.radio("Choose data source", ('Upload Data', 'Use Example Data'))

if data_source == 'Upload Data':
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls', 'tsv'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            data = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.tsv'):
            data = pd.read_csv(uploaded_file, sep='\t')
        st.success("Data uploaded successfully!")
else:
    example_data = st.selectbox("Choose an example dataset", ('titanic', 'tips', 'iris'))
    data = load_example_data(example_data)
    st.success(f"Loaded {example_data} dataset")

if 'data' in locals():
    print_data_info(data)

    st.write("### Select Features and Target")
    columns = data.columns.tolist()
    features = st.multiselect("Select feature columns", columns)
    target = st.selectbox("Select target column", columns)
   
    if target and features:
        problem_type = st.radio("Is this a regression or classification problem?", ('Regression', 'Classification'))

    if st.button("Run Analysis"):
        # Preprocess the data
        data = data.dropna(subset=[target])
        if data[features].isnull().sum().sum() > 0:
            imputer = IterativeImputer()
            data[features] = imputer.fit_transform(data[features])
        
        if problem_type == 'Classification':
            le = LabelEncoder()
            data[target] = le.fit_transform(data[target])
        
        # Scaling numerical features
        scaler = StandardScaler()
        data[features] = scaler.fit_transform(data[features])
        
        # Train-test split
        test_size = st.slider("Select train-test split size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=test_size, random_state=42)
        
        # Model selection
        st.sidebar.write("### Choose Model")
        if problem_type == 'Regression':
            model_choice = st.sidebar.selectbox("Select model", ("Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor"))
        else:
            model_choice = st.sidebar.selectbox("Select model", ("Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier"))

        if st.sidebar.button("Train Model"):
            if problem_type == 'Regression':
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                elif model_choice == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif model_choice == "Support Vector Regressor":
                    model = SVR()
            else:
                if model_choice == "Decision Tree Classifier":
                    model = DecisionTreeClassifier()
                elif model_choice == "Random Forest Classifier":
                    model = RandomForestClassifier()
                elif model_choice == "Support Vector Classifier":
                    model = SVC()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if problem_type == 'Regression':
                st.write("### Regression Evaluation Metrics")
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test)
                r2 = r2_score(y_test)
                st.write(f"MSE: {mse}")
                st.write(f"RMSE: {rmse}")
                st.write(f"MAE: {mae}")
                st.write(f"R2 Score: {r2}")
            else:
                st.write("### Classification Evaluation Metrics")
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
                st.write(f"Accuracy: {accuracy}")
                st.write(f"Precision: {precision}")
                st.write(f"Recall: {recall}")
                st.write(f"F1 Score: {f1}")
                st.write("Confusion Matrix:")
                st.write(cm)


                # Save model
                if st.sidebar.button("Download Model"):
                    pickle.dump(model, open("best_model.pkl", "wb"))
                    st.sidebar.success("Model downloaded successfully!")
    
                # Make predictions
                if st.sidebar.button("Make Prediction"):
                   st.write("Provide input data for prediction")
                   input_data = {}
                   for feature in features:
                       input_data[feature] = st.slider(f"{feature}", float(data[feature].min()), float(data[feature].max()))
        
                   input_df = pd.DataFrame([input_data])
                   input_df[features] = scaler.transform(input_df[features])
                   prediction = model.predict(input_df[features])
                   st.write("Prediction:", prediction)






 #streamlit run c:/Users/ok/Desktop/python_for_data_science/13_streamlit/03_app_two/app.py
 
 #streamlit run c:/Users/ok/Desktop/python_for_data_science/13_streamlit/03_ml_app/app.py