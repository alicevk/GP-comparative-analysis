# Import needed libraries and modules
from codecarbon import EmissionsTracker
import numpy as np
import torch
import gpytorch
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from optuna import create_study
import json


# Fetch dataset from UCI Repository
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.original

# ---------------------------------------------------------------------------- #
#                                PRE-PROCESSING                                #
# ---------------------------------------------------------------------------- #

##### SETTINGS #####
PC_Features = True
Random_Seed = 82024
K_Folds = 10
Max_Iterations = 200
Study = "gpytorch-study"
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

# Define pipeline depending on whether PCA is requested or not
if PC_Features:
    preprocessor = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=12))
    ])

# ---------------------------------------------------------------------------- #
#                              MODEL OPTIMIZATION                              #
# ---------------------------------------------------------------------------- #

# Initiate CodeCarbon to track emissions
tracker = EmissionsTracker('GP gpytorch optimization', log_level='warning')
tracker.start()

# ---------------------------------- OPTUNA ---------------------------------- #

# Create model class
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_X, train_y, likelihood, kernel, lengthscale):
        super(ExactGPModel, self).__init__(train_X, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Kernel setup
        Kernels = {
            'RBF': gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale=lengthscale, ard_num_dims=12)),
            'Matern': gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, lengthscale=lengthscale, ard_num_dims=12)),
            'Periodic': gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(lengthscale=lengthscale, ard_num_dims=12)),
            'Linear': gpytorch.kernels.LinearKernel(),
            'SpectralMixture': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=12)
        }
        
        self.covar_module = Kernels[kernel]
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Function to create models
def create_model_instance(trial, train_X, train_y):
    # Suggest hyperparameters
    kernel = trial.suggest_categorical('kernel', ['RBF', 'Matern', 'Periodic', 'Linear', 'SpectralMixture'])
    lengthscale = X.shape[-1]
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_X, train_y, likelihood, kernel=kernel, lengthscale=lengthscale)

    return likelihood, model, learning_rate

# K_Fold cross validation function
def cross_validate_trial(trial, X, y, roc_auc_only=True):
    kfold = KFold(n_splits=K_Folds, shuffle=True, random_state=Random_Seed)
    roc_aucs, accs = [], []

    for train_idx, test_idx in kfold.split(X):
        # Split data into training and testing sets
        train_X, test_X = X.iloc[train_idx], X.iloc[test_idx]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_idx]
        
        # Preprocess data
        train_X = preprocessor.fit_transform(train_X)
        test_X = preprocessor.transform(test_X)
        
        # Convert to PyTorch tensors
        train_X = torch.tensor(train_X, dtype=torch.float32)
        train_y = torch.tensor(train_y.values, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        test_y = torch.tensor(test_y.values, dtype=torch.float32)        
        
        # Initialize model and likelihood
        likelihood, model, learning_rate = create_model_instance(trial, train_X, train_y)

        # Train model
        model.train()
        likelihood.train()
        
        # Use Adam optimizer
        optimizer = Adam(model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(Max_Iterations):
            optimizer.zero_grad()
            output = model(train_X)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_pred = likelihood(model(test_X))
            pred_probs = test_pred.mean.numpy()
        roc_aucs.append(roc_auc_score(test_y.numpy(), pred_probs))
        accs.append(accuracy_score(test_y.numpy(), (pred_probs > 0.5).astype(int)))

    # Calculate and return results
    acc = np.mean(accs)
    acc_std = np.std(accs)
    roc_auc = np.mean(roc_aucs)
    roc_auc_std = np.std(roc_aucs)

    if roc_auc_only:
        return roc_auc
    else:
        return acc, acc_std, roc_auc, roc_auc_std

# Objective function for Optuna
def objective(trial):    
    roc_auc = cross_validate_trial(trial, X, y)
    return roc_auc

# Create the study and run optimization
study = create_study(
    study_name=Study,
    storage=f"sqlite:///{Study}.db",
    direction="maximize",
    load_if_exists=True,
)
study.optimize(objective, n_trials=Num_Trials)

# Print best trial
best_trial = study.best_trial
print("Best trial:")
print(f"Score: {best_trial.value}")
print("Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Evaluate the best model
acc, acc_std, roc_auc, roc_auc_std = cross_validate_trial(best_trial, X, y, roc_auc_only=False)

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

with open('gpytorch-best-trial.json', 'w') as f:
    json.dump(best_trial_params, f)
    
# Stop emission tracking
_ = tracker.stop()