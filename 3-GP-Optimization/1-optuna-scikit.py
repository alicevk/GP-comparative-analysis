# Import needed libraries and modules
from codecarbon import EmissionsTracker
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.pipeline import Pipeline
from optuna import create_study
import json

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
Max_Iterations = 500
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

# Function to create models
def create_model_instance(trial):
    # Suggest hyperparameters
    kernel_id = trial.suggest_categorical("kernel", ["rbf", "matern"])

    len_scale = trial.suggest_float('length_scale', 1, 1e2, log=True)
    nu = trial.suggest_float('nu', 1e-2, 1e2, log=True)
 
    # Kernel setup
    Kernels = {
        'rbf': RBF(length_scale=len_scale),
        'matern': Matern(length_scale=len_scale, nu=nu),
    }
 
    kernel = Kernels[kernel_id]
    
    parameters = {
        "kernel": kernel,
        "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 10),
        "max_iter_predict": Max_Iterations,
        "random_state": Random_Seed
    }

    print(f'Now trying: {parameters}')

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

# Save best trial parameters to a JSON file
with open('params/scikit-params.json', 'w') as f:
    json.dump(best_trial.params, f)

# Stop emission tracking
_ = tracker.stop

# Save evaluation to an external file
file = 'scores.csv'

results = pd.DataFrame({
    'scikit-optimization': [acc, acc_std, roc_auc, roc_auc_std]
}, index = ['Accuracy', 'Accuracy STD', 'AUC-ROC', 'AUC-ROC STD'])

# Check if file exists
if os.path.exists(file):
    temp = pd.read_csv(file, index_col=0)
    results = pd.concat([temp, results], axis=1)

# Export
results.to_csv(file)