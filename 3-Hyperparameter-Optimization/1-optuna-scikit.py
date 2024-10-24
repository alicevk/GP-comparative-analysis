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
from sklearn.pipeline import Pipeline

# Fetch dataset from UCI Repository
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.original

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #

##### SETTINGS #####
PC_Features = True
Random_Seed = 82024
K_Folds = 10
Max_Iterations = 200
Study = "scikit-study"
Num_Trials = 100
####################

# Drop missing values
df = df.dropna()
df = df.reset_index(drop=True)

# Binarize data
df.loc[df["num"] != 0, "num"] = 1

# Define features and target vectors
X = df.iloc[:,:-1]
y = df['num']

# Separate integer from categorical features
int_features, cat_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],\
['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('int', StandardScaler(), int_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

# ---------------------------------------------------------------------------- #
#                              MODEL OPTIMIZATION                              #
# ---------------------------------------------------------------------------- #

# Initiate CodeCarbon to track emissions
tracker = EmissionsTracker('GP scikit optimization', log_level='warning')
tracker.start()

# ---------------------------------- OPTUNA ---------------------------------- #

# Kernel setup
Kernels = {
    "rbf": 1 * RBF(),
    "dot": 1 * DotProduct(),
    "matern": 1 * Matern(),
    "white": 1 * WhiteKernel(noise_level_bounds=(1e-10, 1e10))
}

# Function to create models
def create_model_instance(trial):
    # Suggest hyperparameters
    kernel_id = trial.suggest_categorical("kernel", ["rbf", "dot", "matern", "white"])
    
    if kernel_id == 'matern':
        nu = trial.suggest_categorical("matern_nu", [0.5, 1.5, 2.5])
        kernel = 1 * Matern(nu=nu)
    elif kernel_id == 'white':
        noise_level = trial.suggest_float("white_noise", 1e-6, 1e2, log=True)
        kernel = 1 * WhiteKernel(noise_level=noise_level)
    else:
        kernel = Kernels[kernel_id]
    
    parameters = {
        "kernel": kernel,
        "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 10),
        "max_iter_predict": Max_Iterations,
        "random_state": Random_Seed
    }

    model = GaussianProcessClassifier(**parameters)
    
    return model

# Objective function for Optuna
def objective(trial):
    model = create_model_instance(trial)
    
    # Define pipeline depending on whether PCA is requested or not
    if PC_Features:
        steps = [
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=12)),
            ('model', model)
        ]
    else:
        steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
        
    pipeline = Pipeline(steps)
        
    roc_auc = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=K_Folds)
    return roc_auc.mean()

# Create the study and run optimization
study = create_study(
    study_name=Study,
    storage=f"sqlite:///{Study}.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(lambda trial: objective(trial), n_trials=Num_Trials)

# Print best trial
best_trial = study.best_trial
print("Best trial:")
print(f"Score: {best_trial.value}")
print("Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Evaluate best model
model = create_model_instance(best_trial)
cv_results = cross_validate(model, X, y, scoring=['accuracy', 'roc_auc'], cv=K_Folds)

# Calculate and display results
acc = np.mean(cv_results['test_accuracy'])
acc_std = np.std(cv_results['test_accuracy'])
roc_auc = np.mean(cv_results['test_roc_auc'])
roc_auc_std = np.std(cv_results['test_roc_auc'])

print(f"Accuracy: {acc:.4f} ± {acc_std:.4f}")
print(f"AUC-ROC: {roc_auc:.4f} ± {roc_auc_std:.4f}")

# Save best trial parameters and evaluation to a JSON file
best_trial_params = {
    'params': best_trial.params,
    'evaluation':{
        'accuracy': acc,
        'accuracy STD': acc_std,
        'ROC AUC': roc_auc,
        'ROC AUC STD': roc_auc_std
}}

with open('scikit-best-trial.json', 'w') as f:
    json.dump(best_trial_params, f)

# Stop emission tracking
_ = tracker.stop