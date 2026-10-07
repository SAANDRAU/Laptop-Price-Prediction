# Laptop Price Prediction 

A machine learning project that predicts the estimated price of a laptop based on its specifications. The project includes data preprocessing, exploratory data analysis, regression model comparison, hyperparameter tuning, model evaluation, and a Streamlit-based prediction interface.

## 📌 Project Overview

The goal of this project is to build a machine learning model that can estimate laptop prices from specifications such as:

- Brand
- Processor brand and name
- Processor generation
- RAM
- RAM type
- SSD
- HDD
- Operating system
- Operating system bit
- Graphics card
- Weight
- Warranty
- Touchscreen
- MS Office availability
- Rating
- Number of ratings

The project was developed using Python and Scikit-learn, with a Streamlit interface for making predictions.

##  Technologies Used

- **Python**
- **Pandas** – data manipulation and preprocessing
- **NumPy** – numerical operations
- **Matplotlib** – data visualization
- **Seaborn** – exploratory data analysis and visualization
- **Scikit-learn** – machine learning and model evaluation
- **XGBoost** – regression model comparison
- **Streamlit** – interactive prediction interface
- **Pickle** – model serialization

##  Project Workflow

### 1. Data Loading

The laptop dataset was loaded using Pandas and inspected to understand:

- Dataset structure
- Data types
- Number of records and features
- Missing values
- Duplicate records
- Statistical characteristics

### 2. Data Cleaning

The dataset was cleaned by:

- Checking and removing duplicate records
- Detecting price outliers using the IQR method
- Handling outlier values
- Converting specifications such as RAM, SSD, HDD and graphics card capacity into numerical values

For example, values such as `8GB` were converted into numerical values such as `8`.

### 3. Exploratory Data Analysis

Exploratory analysis was performed using:

- Box plots
- Histograms
- Correlation analysis
- Descriptive statistics

This helped understand the distribution of laptop prices and relationships between numerical features.

### 4. Feature Encoding

Categorical features were converted into numerical representations using `LabelEncoder`.

Features encoded include:

- Brand
- Processor brand
- Processor name
- Processor generation
- RAM type
- Operating system
- OS bit
- Weight
- Warranty
- Touchscreen
- MS Office
- Rating

### 5. Train-Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

A `random_state` of 42 was used to make the split reproducible.

### 6. Feature Scaling

`StandardScaler` was used to scale the input features before training the regression models.

### 7. Machine Learning Models

Several regression algorithms were compared, including:

- Lasso Regression
- Ridge Regression
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regression
- Random Forest Regressor
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Model performance was compared using **R² score** with 5-fold cross-validation.

### 8. Hyperparameter Tuning

`GridSearchCV` was used to find suitable hyperparameters for the Gradient Boosting Regressor.

The parameters explored included:

- Number of estimators
- Learning rate
- Maximum depth
- Minimum samples split
- Minimum samples leaf

### 9. Model Evaluation

The final model was evaluated on the test dataset using:

- **Mean Absolute Error (MAE)**
- **R² Score**

### 10. Model Saving

The trained model and preprocessing components were saved using Pickle.

The saved components include:

- Trained regression model
- Label encoders
- Standard scaler

This allows the trained model to be reused without retraining it every time.

##  Streamlit Application

A Streamlit-based interface was created to allow users to enter laptop specifications and receive an estimated laptop price.

The application:

1. Takes laptop specifications as input
2. Applies the saved preprocessing steps
3. Uses the trained machine learning model
4. Predicts the estimated price
5. Displays the result to the user

##  Project Structure

```text
Laptop-Price-Prediction/
│
├── Laptop_price_prediction.ipynb
├── trial.py
├── laptop_price_model.pkl
├── laptopPrice.csv
└── README.md
