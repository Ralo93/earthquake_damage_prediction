# Earthquake Damage Prediction

This repository contains our team's approach to the Earthquake Damage Prediction competition hosted on DrivenData.org. 

### The challenge and the results

- Submit a solution within 2 days, while managing a team.
- We were able to score the highest micro-F1-score out of all participants in the mini-competition hosted by DataScienceRetreat.
- Being only 1% below rank 1 in the overall competition, I would consider this a success, as we submitted only 6 solutions.

## Project Overview

Prediction of damage caused by an earthquake.
We used a dataset by drivendata.org, which consists out of a collection of datapoints that represent buildings in an earthquake stroke area.
The dataset is from 2015 and contains roughly 1/4million datapoints.

Our approach consisted of data cleaning, preprocessing and trying different supervised learning algorithms. Further it included hyperparameter tuning with cross-validation using bayesian search.

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
5. Crossvalidation on the Auto-Encoding parameter dimensions (5, 10, 15, 20)
6. Additional Auto-Encoding after the BaseEncoding to reduce dimensionality of GeoID features to 5
7. Transforming the age column and make a new feature for very old buildings
8. BaseEncoding of categorical features with base of 3
9. Automatic hyperparameter tuning with bayesian search (hyperopt package)

### Experiment Results

- LGBM: Average Micro F1 score
  - Train: 0.7974
  - Validation: 0.7443
 
![image](https://github.com/user-attachments/assets/de47b871-ec0f-464a-a851-d42f4ce8644a)



- XGB: Average F1 score
  - Train: 0.8024
  - Validation: 0.7419

![image](https://github.com/user-attachments/assets/cfb66e84-918c-410c-aa1a-e74eed66e17a)

Other Experiments which did not improve results:
- Feature Splitting on the "has_superstructure..." columns
- Clustering on area and height
- Autoencoding of interaction features of geoIDs
- Dropping of columns which seemed not important
- Sequential models: Use two binary classificators to predict 1 | 2,3 then another model 2 | 3
- Replace geo-locations with few instances with another number (i.e., an numerical "other").
- Target-encoding geolocation 

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
