# Football Player Performance Prediction

This project predicts the overall performance rating of football players using a SQLite database containing player attributes and statistics. Three models are developed and compared: a Neural Network, Random Forest, and XGBoost. The goal is to identify the most effective model for predicting player performance based on features such as physical attributes, skills, and work rates.

## Project Overview
This project leverages a football player dataset to predict overall ratings using three machine learning models:
- **Neural Network (NN)**: A deep learning model built with TensorFlow and Keras.
- **Random Forest**: An ensemble model using decision trees, implemented with scikit-learn for robust predictions.
- **XGBoost**: A gradient boosting model, implemented with the XGBoost library, known for high performance in regression tasks.

The models are evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE). Visualizations, including feature correlation heatmaps and training loss curves, provide insights into the data and model performance. A comparison analysis evaluates the trade-offs between the models' accuracy, interpretability, and computational efficiency.

## Dataset
Link: [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer)

## Methodology
-----------

The project follows these steps:

1.  **Data Loading**:
    
    *   Load the Player and Player\_Attributes tables from the SQLite database using pandas and sqlite3.
        
2.  **Data Exploration**:
    
    *   Inspect the structure and content of the Player and Player\_Attributes DataFrames.
        
3.  **Preprocessing**:
    
    *   Drop irrelevant columns (e.g., player names, IDs). 
    *   Extract birth year, month, and day from the birthday column.  
    *   Aggregate player statistics by averaging numeric attributes and handling categorical features (e.g., preferred foot, attacking/defensive work rates).  
    *   Encode categorical variables (e.g., one-hot encoding for work rates, binary encoding for preferred foot).  
    *   Scale features using StandardScaler.  
    *   Split data into training (70%) and test (30%) sets.
        
4.  **Data Visualization**:
    
    *   Plot the distribution of overall ratings to understand the target variable.
    *   Create a correlation heatmap for the top 15 features most correlated with the overall rating.
        
5.  **Model Development**:
    
    *   **Neural Network**:
        *   Built using Keras with an input layer, three hidden layers (128, 64, 32 units) with ReLU activation, and a single output unit.
    *   **Random Forest**:
        *   Implemented using sklearn.ensemble.RandomForestRegressor with default or tuned parameters (e.g., number of trees, max depth). 
    *   **XGBoost**:   
        *   Implemented using xgboost.XGBRegressor with default or tuned parameters (e.g., learning rate, max depth).
            
6.  **Model Training**:
    
    *   Train the Neural Network for 50 epochs with a batch size of 64, using the Adam optimizer and Mean Squared Error loss.  
    *   Train Random Forest and XGBoost models on the same training data.
        
7.  **Model Evaluation**:
    
    *   Evaluate all models on the test set using MSE and MAE.  
    *   Compare predicted vs. actual ratings for a sample of test data.
        
8.  **Comparison Analysis**:
    
    *   Compare the Neural Network, Random Forest, and XGBoost models based on test set MSE and MAE.    
    *   Analyze trade-offs, such as the Neural Network’s flexibility, Random Forest’s interpretability, and XGBoost’s computational efficiency.
