# Earthquake Damage Prediction

This repository contains our team's approach to the Earthquake Damage Prediction competition hosted on DrivenData.org.

## Team Members

- Alan Tump
- Shan Jiang
- Raphael von Lottner

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

## Setup (coming soon)

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

1. LightGBMClassifier
2. XGBoostClassifier
3. RandomForrestClassifier

[Brief description of each model and why it was chosen]

## Experiments

Our experimental process included:

0. Kept all data except building_id
1. Outlier removal on the 97th percentile
2. Unskew right skewed data
4. BaseEncoding of geoIDs with a base of 3
5. Additional Auto-Encoding after the BaseEncoding to reduce dimensionality of GeoID features to 5
6. Transforming the age column and make a new feature for very old buildings
7. BaseEncoding of categorical features with base of 3
8. Automatic hyperparameter tuning with bayesian search (hyperopt package)

### Experiment Results

- AT: Average F1 score
  - Train: 0.76
  - Test: 0.73
  (See AT_Exp_XG_BiasVar.ipynb for details)

Other Experiments which did not improve results:
- Feature Splitting on the "has_superstructure..." columns
- Clustering on area and height
- Autoencoding of interaction features of geoIDs
- Dropping of columns which seemed not important
- Sequential models: Use two binary classificators to predict 1 | 2,3 then another model 2 | 3

## Results
![image](https://github.com/user-attachments/assets/d692bf77-cc40-4b4f-9a5f-16734166c145)
![image](https://github.com/user-attachments/assets/0c700040-ab00-47a5-b43c-afc32e38504c)

## Reproduction

To reproduce our results: (coming soon)

1. [Step 1]
2. [Step 2]
3. [Step 3]

## Resources

(coming soon)

## Future Work

[Ideas for improving the model or extending the project]
