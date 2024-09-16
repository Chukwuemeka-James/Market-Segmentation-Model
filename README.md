# Market Segmentation Model

This repository contains an end to end Market Segmentation project utilizing **KMeans Clustering** and **Decision Tree Classification**. The goal of the project is to segment customers into clusters based on various financial attributes and then predict these segments using a decision tree classifier.

## Project Overview

Market segmentation is the process of dividing a customer base into distinct groups that share common characteristics. This helps in targeting specific customer groups for marketing and other strategic decisions. In this project, we have used unsupervised learning (KMeans Clustering) to identify customer segments, and supervised learning (Decision Tree) to predict customer clusters.

### Dataset

The dataset used for this project contains customer data with attributes such as:

- Balance
- Purchases
- Cash Advance
- Credit Limit
- Payments
- Minimum Payments
- And more...

Missing values in the dataset are handled by filling in the mean values for missing columns like `MINIMUM_PAYMENTS` and `CREDIT_LIMIT`.

### Key Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **Numpy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine Learning algorithms and preprocessing
- **Joblib & Pickle**: Model serialization
- **Warnings**: For ignoring unnecessary warnings

## Project Workflow

### 1. Data Preprocessing

- **Standard Scaling**: Standardizes the dataset to have mean=0 and standard deviation=1.
- **Principal Component Analysis (PCA)**: Reduces the dataset to two principal components for visualization purposes.
  
### 2. Clustering Using KMeans

- **Elbow Method**: Used to find the optimal number of clusters.
- **KMeans Algorithm**: Segments the customers into clusters based on their financial attributes.
- **Cluster Analysis**: Visualized using scatter plots and Kernel Density Estimation (KDE) plots for each cluster.

### 3. Decision Tree Classifier

- **Model Training**: A decision tree classifier is trained using the customer clusters as the target variable.
- **Model Evaluation**: Evaluated using confusion matrix and classification report.
- **Model Serialization**: The decision tree model is saved for future predictions using Pickle.

### 4. Model Performance

- The accuracy of the decision tree classifier is evaluated on the test dataset and reported.

## Key Features

- **Clustering (KMeans)**: Groups customers into 4 segments.
- **Dimensionality Reduction (PCA)**: Used to visualize the clusters.
- **Classification (Decision Tree)**: Predicts the cluster/segment to which a customer belongs.
- **Model Saving and Loading**: Using `joblib` and `pickle` for reusability of models.