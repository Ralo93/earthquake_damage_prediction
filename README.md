# Earthquake Damage Prediction

This repository contains our team's approach to the Earthquake Damage Prediction competition hosted on DrivenData.org.

## Team Members

- Alan Tump
- Shan Jiang
- Raphael von Lotter

## Project Overview

Prediction of damage caused by an earthquake.
We used a dataset by drivendata.org, which consists out of a collection of datapoints that represent buildings in an earthquake stroke area.
The dataset is from 2015 and contains roughly 1/4million datapoints.

Our approach consisted of data cleaning, preprocessing and trying different supervised learning algorithms. Further it included cross-validation and hyperparameter tuning using bayesian search.

## Table of Contents

- [Setup](#setup)
- [Data](#data)
- [Approach](#approach)
- [Models](#models)
- [Experiments](#experiments)
- [Results](#results)
- [Reproduction](#reproduction)
- [Future Work](#future-work)

## Setup

To get started with this project:

1. Clone the repository:
   ```
   git clone [repository URL]
   ```
2. Create your own branch:
   ```
   git checkout -b [your-branch-name]
   ```
3. Set up your environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

TODO: Create requirements.txt file

## Data

[Description of the dataset, including its source, features, and target variable]

## Approach

[Overview of our approach to solving the problem]

## Models

We experimented with several models, including:

1. [Model 1]
2. [Model 2]
3. [Model 3]

[Brief description of each model and why it was chosen]

## Experiments

Our experimental process included:

1. [Experiment 1]
2. [Experiment 2]
3. [Experiment 3]

### Experiment Results

- AT: Average F1 score
  - Train: 0.76
  - Test: 0.73
  (See AT_Exp_XG_BiasVar.ipynb for details)

[Add more experiment results as needed]

## Results

[Summary of our best results and performance metrics]

## Reproduction

To reproduce our results:

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Resources

[Insert here: list of additional resources, datasets, or tools used]

## Future Work

[Ideas for improving the model or extending the project]
