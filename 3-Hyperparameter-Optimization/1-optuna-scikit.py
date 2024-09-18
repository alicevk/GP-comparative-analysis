# Import needed libraries and modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import optuna
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from optuna import create_study, Trial

# Fetch dataset from UCI Repository
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.original

# ---------------------------------------------------------------------------- #
#                                PRE-PROCESSING                                #
# ---------------------------------------------------------------------------- #

# --------------------------------- SETTINGS --------------------------------- #
Normalize = False
PC_Features = True
Test_Size = 0.2
Random_Seed = 82024
Torch = False
Num_trials = 1000
Num_folds = 10
Study_name = "gp_scikit_heart_1"
Score = "roc_auc"  # Or "f1"

# Kernel setup
Kernels = {
    "rbf": 1 * RBF(),
    "dot": 1 * DotProduct(),
    "matern": 1 * Matern(),
    "quad": 1 * RationalQuadratic(),
    "white": 1 * WhiteKernel(),
}

# ------------------------------- DATA HANDLING ------------------------------ #
# Drop missing values
df = df.dropna()
df = df.reset_index(drop=True)

# Binarize data
df.loc[df["num"] != 0, "num"] = 1

# Define features and target vectors
X = df.iloc[:,:-1]
y = df['num']

# Normalize if requested
if (Normalize) or (PC_Features):
    int_features, cat_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],\
    ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    preprocessor = ColumnTransformer(
    transformers=[
        ('int', StandardScaler(), int_features),
        ('cat', OneHotEncoder(), cat_features)
    ])
    X = preprocessor.fit_transform(X)
else:
    X = X.values

# Apply PCA if requested
if PC_Features:
    pca = PCA(n_components=12)
    X = pca.fit_transform(X)

# Split train and test data
index = list(range(y.size))
train_index, test_index = train_test_split(index, test_size=Test_Size, random_state=Random_Seed)

train_X = X[train_index]
train_y = y.loc[train_index].values

test_X = X[test_index]
test_y = y.loc[test_index].values

# Convert to torch tensor if requested
if Torch:
    train_X, train_y, test_X, test_y = torch.tensor(train_X),
    torch.tensor(train_y), torch.tensor(test_X), torch.tensor(test_y)

# ---------------------------------- OPTUNA ---------------------------------- #
# Function to create model instances
def create_instance_model(trial):
    """Create an instance of the model."""
    kernel_id = trial.suggest_categorical("kernel", ["rbf", "white", "dot", "matern", "quad"])

    parameters = {
        "kernel": Kernels[kernel_id],
        "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 10),
        "max_iter_predict": trial.suggest_int("max_iter_predict", 50, 1000, log=True),
        "random_state": Random_Seed,
    }

    model = GaussianProcessClassifier(**parameters)
    return model

# Objective function for Optuna
def objective_function(trial, X, y, Num_folds=Num_folds, random_state=Random_Seed):
    """Optuna's objective function"""
    model = create_instance_model(trial)

    metrics = cross_val_score(model, X, y, scoring=Score, cv=Num_folds)
    return metrics.mean()

# Create the study with Optuna
study = create_study(
    study_name=Study_name,
    storage=f"sqlite:///{Study_name}.db",
    direction="maximize",
    load_if_exists=True,
)

study.optimize(lambda trial: objective_function(trial, train_X, train_y), n_trials=Num_trials)

# Save and display the best results
trialdf = study.trials_dataframe()
trialdf.to_csv("trial_df.csv", index=False)

best_trial = study.best_trial
print(best_trial)

# Train and evaluate the final model
model = create_instance_model(best_trial)
model.fit(train_X, train_y)

# Test the model
y_pred = model.predict(test_X)
pred_probs = model.predict_proba(test_X)

# Model evaluation
acc = accuracy_score(test_y, y_pred)
roc_auc = roc_auc_score(test_y, pred_probs[:, 1])

print(f"Accuracy: {acc:.2f}")
print(f"AUC-ROC: {roc_auc:.2f}")
