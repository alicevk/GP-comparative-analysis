from setuptools import setup, find_packages

setup(
    name='GP_comparative_analysis',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'seaborn',
        'matplotlib',
        'numpy',
        'scikit-learn',
        'torch',
        'gpytorch',
        'pyro-ppl',
        'optuna',
        'codecarbon',
        'ucimlrepo'
    ],
    extras_require={
        'dev': [
            'jupyter',
            'pytest'
        ]
    },
)

# pode estar errado!
# PyTorch tem que fazer a instalação na mão!