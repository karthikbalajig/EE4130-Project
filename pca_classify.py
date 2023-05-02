import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load the data
breast_cancer_dataset = load_breast_cancer()
df = pd.DataFrame(breast_cancer_dataset.data,
                  columns=breast_cancer_dataset.feature_names)


# Extract the data, standardize it and split into train and test data
X = df.iloc[:, :]
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

y = breast_cancer_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Perform Logistic regression on all features
logistic_regression = LogisticRegression(max_iter=5000)
logistic_regression.fit(X_train, y_train)

# Score the model
print(
    f'Score of Vanilla Logistic Regression on {X_train.shape[1]} features: {logistic_regression.score(X_test, y_test)}')

# Perform PCA to reduce dimensions
# We perform the PCA with 5 components
pca = PCA(n_components=5)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the model
logistic_regression_pca = LogisticRegression(max_iter=5000)
logistic_regression.fit(X_train_pca, y_train)

# Score the model
print(f'Score of Logistic Regression Score (PCA) on {X_train_pca.shape[1]} features:',
      logistic_regression.score(X_test_pca, y_test))
