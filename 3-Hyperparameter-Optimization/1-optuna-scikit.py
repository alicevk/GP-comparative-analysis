# Import needed libraries and modules
from codecarbon import EmissionsTracker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import optuna
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from optuna import create_study, Trial
import json

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
Num_trials = 100
Study_name = "scikit-study"
Score = "roc_auc"
Num_iterations = 50

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
    train_X, train_y, test_X, test_y = torch.tensor(train_X), torch.tensor(train_y), torch.tensor(test_X), torch.tensor(test_y)

# ---------------------------------------------------------------------------- #
#                                 OPTIMIZATION                                 #
# ---------------------------------------------------------------------------- #
# Initiate CodeCarbon to track emissions
tracker = EmissionsTracker('GP scikit optimization', log_level='warning')
tracker.start()

# ---------------------------------- OPTUNA ---------------------------------- #
# Function to create model instances
def create_instance_model(trial):
    """Create an instance of the model."""
    kernel_id = trial.suggest_categorical("kernel", ["rbf", "white", "dot", "matern", "quad"])

    parameters = {
        "kernel": Kernels[kernel_id],
        "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 10),
        # "max_iter_predict": trial.suggest_int("max_iter_predict", 50, 500, log=True),
        "random_state": Random_Seed
    }

    model = GaussianProcessClassifier(**parameters)
    return model

# Objective function for Optuna
def objective_function(trial):
    """Optuna's objective function"""
    model = create_instance_model(trial)

    metrics = cross_val_score(model, X, y, scoring=Score)
    return metrics.mean()

# Create the study with Optuna
study = create_study(
    study_name=Study_name,
    storage=f"sqlite:///{Study_name}.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(lambda trial: objective_function(trial), n_trials=Num_trials)

# Print best trial
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Evaluate the best model
model = create_instance_model(best_trial)
acc, roc_auc = cross_validate(model, X, y, scoring=['accuracy', 'roc_auc'])

print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Save best trial parameters and evaluation to a JSON file
best_trial_params = {
    'params': best_trial.params,
    'evaluation':{
        'accuracy': acc,
        'roc_auc': roc_auc
}}

with open('scikit-best-trial.json', 'w') as f:
    json.dump(best_trial_params, f)
    
tracker.stop