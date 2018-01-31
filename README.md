# UCI Madelon Dataset Project

### Domain and Data

Data: [Madelon](https://archive.ics.uci.edu/ml/datasets/Madelon)

"*Abstract:* MADELON is an artificial dataset, which was part of the NIPS 2003 feature selection challenge. This is a two-class classification problem with continuous input variables. The difficulty is that the problem is multivariate and highly non-linear."



### Problem Statement

1. identifying relevant features. 
2. generating predictions from the model. 


### Tasks

#### Data Manipulation

You should do substantive work on at least six subsets of the data. 

- Subset of 10% of the data from the UCI Madelon set
- All of the data from the UCI Madelon set
- Subset of 10% of the data from the Madelon set made available by an instructor (Joshua Cook)
- All of the data from the Madelon set made available by an instructor (Joshua Cook)

##### Prepared Report

Your report should:

1. be a pdf
2. include EDA of each subset 
   - EDA needs may be different depending upon subset or your approach to a solution
3. present results from Step 1: Benchmarking
4. present results from Step 2: Identify Salient Features
5. present results from Step 3: Feature Importances
6. present results from Step 4: Build Model

##### Jupyter Notebook, EDA 

- perform EDA on each set as you see necessary

##### Jupyter Notebook, Step 1 - Benchmarking
- build pipeline to perform a naive fit for each of the base model classes:
	- logistic regression
	- decision tree
	- k nearest neighbors
	- support vector classifier
- in order to do this, you will need to set a high `C` value in order to perform minimal regularization, in the case of logistic regression and support vector classifier.

##### Jupyter Notebook, Step 2 - Identify Features
- Build feature selection pipelines using at least three different techniques
- **NOTE**: these pipelines are being used for feature selection not prediction

##### Jupyter Notebook, Step 3 - Feature Importance
- Use the results from step 2 to discuss feature importance in the dataset
- Considering these results, develop a strategy for building a final predictive model
- recommended approaches:
    - Use feature selection to reduce the dataset to a manageable size then use conventional methods
    - Use dimension reduction to reduce the dataset to a manageable size then use conventional methods
    - Use an iterative model training method to use the entire dataset
   
##### Jupyter Notebook, Step 4 - Build Model
- Implement your final model
- (Optionally) use the entire data set

---

### Requirements

- Many Jupyter Notebooks
- A written report of your findings that detail the accuracy and assumptions of your model.

---

### Suggestions

- Document **everything**.

