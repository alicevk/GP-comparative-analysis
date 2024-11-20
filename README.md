# Análise comparativa de algoritmos de Processo Gaussiano para classificação de dados de cardiopatia

Projeto final do Bacharelado em Ciência e Tecnologia da Ilum - Escola de Ciência.

## Resumo
O Aprendizado de Máquina (*Machine Learning*, ML) tem revolucionado a área médica ao tornar possível o desenvolvimento de ferramentas diagnósticas precisas e eficientes em uma escala muito maior do que os métodos tradicionais. Nesse sentido, o presente estudo visa comparar três diferentes implementações do método de Processos Gaussianos (*Gaussian Processes*, GP) -- assim como outros algoritmos de ML -- para a classificação de dados referentes à Doença Arterial Coronariana (DAC). Foi realizada uma análise comparativa abrangente entre os diferentes modelos, considerando não apenas as características técnicas de cada um, mas também aspectos relacionados ao impacto ambiental da Inteligência Artificial e a respecussão disso na própria área da saúde, motivação inicial da pesquisa.

---

# Comparative Analysis of Gaussian Process Algorithms for Heart Disease Data Classification

## Abstract
Machine Learning (ML) has revolutionized the medical field by enabling the development of precise and efficient diagnostic tools at a much larger scale than traditional methods. In this context, the present study aims to compare three different implementations of the Gaussian Processes (GP) method—as well as other ML algorithms—for the classification of data related to Coronary Artery Disease (CAD). A comprehensive comparative analysis was performed between the different models, considering not only the technical characteristics of each but also aspects related to the environmental impact of Artificial Intelligence and its repercussions in the healthcare field, the initial motivation of this research.

---

## Repository Guide

This repository is organized in different folders, dividing each step of the project.

| Folder | Content Description |
| ------ | ------------------- |
| [1 - EDA and Preprocessing](1-EDA-and-Preprocessing) | * Dataset overview and exploratory data analysis <br> * Dimensionality reduction |
| [2 - GP Implementation](2-GP-Implementation) | * SciKit-Learn implementation <br> * GPytorch implementation <br> * Pyro implementation |
| [3 - GP Optimization](3-GP-Optimization) | * Optuna + SciKit-Learn model <br> * Optuna + GPytorch model |
| [4 - Alternative Models](4-Alternative-Models) | * Lazy learners <br> * Eager learners |
| [5 - SHAP Values](5-SHAP-Values) | * SHAP Analysis: SciKit-Learn model <br> * SHAP Analysis: lazy learners <br> * SHAP Analysis: eager learners |
| [6 - General Comparison](6-General-Comparison) | * Emission visualization <br> * Score comparison |


Both [`requirements.txt`](requirements.txt) and [`setup.py`](setup.py) files have been included to promote easier access and reproductibility, allowing th use of either of the following commands to set up the environment:

* `pip install -r requirements.txt`

* `python3 -u setup.py install`


<!-- ## Table of Contents
1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Discussion](#discussion)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction


## Methodology


## Results


## Discussion


## Conclusion


## References


---

## License (?)


## Acknowledgements
-->