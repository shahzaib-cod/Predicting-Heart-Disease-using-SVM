import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import os

#Dataset
def load_and_explore_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, header=None, names=columns)
    df = df.replace('?', np.nan)  
    df.dropna(subset=['target'], inplace=True)  
    df['target'] = df['target'].astype(int)
    return df

# Data Preparation
def preprocess_data(df):
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        df[col] = df[col].astype(float)  

    scaler = StandardScaler()
    features = df.drop('target', axis=1)
    features_scaled = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled['target'] = df['target']
    return df_scaled, scaler

#Exploratory Data Analysis (EDA)
def perform_eda(df_scaled):
    df_scaled.hist(bins=20, figsize=(15, 10))
    plt.suptitle('Histograms of Scaled Features')
    plt.show()
    
    plt.figure(figsize=(15, 10))
    df_scaled.boxplot()
    plt.xticks(rotation=90)
    plt.title('Boxplots of Scaled Features')
    plt.show()
    
    corr_matrix = df_scaled.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    sns.pairplot(df_scaled, hue='target')
    plt.suptitle('Scatter Plot Matrix of Features Colored by Target', y=1.02)
    plt.show()

# Feature Engineering and Selection
def feature_engineering_selection(df_scaled):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df_scaled.drop('target', axis=1))
    y = df_scaled['target']

    selector = SelectKBest(score_func=f_classif, k=13)
    X_new = selector.fit_transform(X_imputed, y)
    selected_features = df_scaled.drop('target', axis=1).columns[selector.get_support(indices=True)]
    print("Selected Features:", selected_features)
    return X_new, y, selected_features

#Model Selection and Optimization
def model_selection_optimization(X_new, y):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    
    svm = SVC()
    svm.fit(X_train, y_train)
    
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    
    best_svm = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)
    return best_svm, X_train, X_test, y_train, y_test

#  Model Evaluation and Interpretation
def evaluate_model(best_svm, X_test, y_test, X_new, y):
    y_pred = best_svm.predict(X_test)
    
    disp = ConfusionMatrixDisplay.from_estimator(best_svm, X_test, y_test)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()
    
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    
    train_sizes, train_scores, test_scores = learning_curve(best_svm, X_new, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
    
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.show()
    
    return accuracy


#  Decision Boundary Plot
def plot_decision_boundary(best_svm, X, y):
    
    feature1_index = 0  
    feature2_index = 1  
    
    X_subset = X[:, [feature1_index, feature2_index]]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
    
   
    svm_subset = SVC()
    svm_subset.fit(X_train, y_train)
    
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = svm_subset.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y, edgecolors='k', marker='o')
    plt.title('Decision Boundary Plot')
    plt.xlabel('Feature {}'.format(feature1_index))
    plt.ylabel('Feature {}'.format(feature2_index))
    plt.show()


#Web-based GUI
from flask import Flask, request, render_template, redirect, url_for
import pickle

def create_app(best_svm, scaler, accuracy):
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/start')
    def start():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Convert form values to floats, handling non-numeric inputs
            features = [float(x) if x.strip() != '' else 0.0 for x in request.form.values()]
            features_scaled = scaler.transform([features])
            prediction = best_svm.predict(features_scaled)
            return render_template('index.html', prediction_text='Heart Disease Prediction: {}'.format(prediction[0]), accuracy=accuracy)
        except ValueError as e:
            # Handle the case where a non-numeric input is provided
            return render_template('index.html', prediction_text='Invalid input! Please provide numeric values.', accuracy=accuracy)

    return app

