# Import needed libraries and modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import optuna
from optuna import create_study, Trial
import json
import gpytorch
from torch.optim import Adam


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
Torch = True
Num_trials = 1000
Study_name = "gpytorch-study"
Score = "roc_auc"
Num_iterations = 100


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
    
# Convert to torch tensor if requested
if Torch:
    X = torch.tensor(X)
    y = torch.tensor(y).double()


# ---------------------------------- OPTUNA ---------------------------------- #
# Define GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, lengthscale):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Kernel setup
        Kernels = {
            'RBF': gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale=lengthscale, ard_num_dims=X.shape[-1])),
            'Matern': gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, lengthscale=lengthscale, ard_num_dims=X.shape[-1])),
            'Periodic': gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(lengthscale=lengthscale, ard_num_dims=X.shape[-1])),
            'Linear': gpytorch.kernels.LinearKernel(),
            'SpectralMixture': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=X.shape[-1])
            }
        
        self.covar_module = Kernels[kernel]
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def create_instance_model(trial, train_x, train_y):
    # Suggest hyperparameters
    kernel = trial.suggest_categorical('kernel', ['RBF', 'Matern', 'Periodic', 'Linear', 'SpectralMixture'])
    lengthscale = X.shape[-1]
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    model = ExactGPModel(train_x, train_y, likelihood, kernel=kernel, lengthscale=lengthscale)

    return likelihood, model, learning_rate


def cross_validate_trial(trial, X, y, training_iterations=Num_iterations, roc_auc_only=True):
    kf = KFold(shuffle=True, random_state=Random_Seed)
    roc_aucs = []
    accuracies = []

    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]

        likelihood, model, learning_rate = create_instance_model(trial, train_x, train_y)

        # Train the model
        model.train()
        likelihood.train()
        
        optimizer = Adam([{'params': model.parameters()}], lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(training_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_pred = likelihood(model(test_x))
            pred_probs = test_pred.mean.numpy()
            roc_auc = roc_auc_score(test_y.numpy(), pred_probs)
            accuracy = accuracy_score(test_y.numpy(), (pred_probs > 0.5).astype(int))
            
            roc_aucs.append(roc_auc)
            accuracies.append(accuracy)

    # Calculate mean metrics
    mean_roc_auc = np.mean(roc_aucs)
    mean_accuracy = np.mean(accuracies)

    if roc_auc_only:
        return mean_roc_auc
    else:
        return mean_accuracy, mean_roc_auc

# Objective function for Optuna
def objective(trial):    
    roc_auc = cross_validate_trial(trial, X, y)
    return roc_auc

# Create the study and run optimization
study = create_study(
    study_name=Study_name,
    storage=f"sqlite:///{Study_name}.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=Num_trials)

# Print the best trial
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Evaluate the best model
acc, roc_auc = cross_validate_trial(best_trial, X, y, roc_auc_only=False)

print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Save best trial parameters and evaluation to a JSON file
best_trial_params = {
    'params': best_trial.params,
    'evaluation':{
        'accuracy': acc,
        'roc_auc': roc_auc
}}

with open('gpytorch-best-trial.json', 'w') as f:
    json.dump(best_trial_params, f)