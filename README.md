# Earthquake Damage Prediction
<p align="center"> <img src="https://github.com/user-attachments/assets/f7475789-6abb-4b2f-9ff0-035835c9c0c5" alt="Earthquake Damage"> </p> <p align="center"> <em>Image source: drivendata.org.</em> </p>
This repository contains our team's approach to the Earthquake Damage Prediction competition hosted on DrivenData.org. 

### The challenge and the results

- Submit a solution within 2 days, while managing a team.
- We were able to score the highest micro-F1-score out of all participants in the mini-competition hosted by DataScienceRetreat.
- Being less than 1% below rank 1 in the overall competition, I would consider this a success, as we submitted only 6 solutions.

## Project Overview

Prediction of damage caused by an earthquake.
We used a dataset by drivendata.org, which consists out of a collection of datapoints that represent buildings in an earthquake stroke area.
The dataset is from 2015 and contains roughly 1/4million datapoints.

Our approach consisted of EDA, data cleaning, preprocessing and trying different supervised learning algorithms. Further it included hyperparameter tuning with cross-validation using bayesian search.

As time was a constraint, we worked mainly in jupter labs (I know, not the perfect IDE but for a quick prototyping and EDA etc. actually a good fit in my opinion).


## Table of Contents

- [Data](#data)
- [Approach](#approach)
- [Models](#models)
- [Experiments](#experiments)
- [Results](#results)
- [Reproduction](#reproduction)
- [Future Work](#future-work)


## Data
A first glimpse at the data gave us the following information:

- The dataset consists of 260601 rows and 40 columns, one of which is the target, called "damage_grade".

### The columns:

'building_id', 'geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id',
       'count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage',
       'land_surface_condition', 'foundation_type', 'roof_type',
       'ground_floor_type', 'other_floor_type', 'position',
       'plan_configuration', 'has_superstructure_adobe_mud',
       'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag',
       'has_superstructure_cement_mortar_stone',
       'has_superstructure_mud_mortar_brick',
       'has_superstructure_cement_mortar_brick', 'has_superstructure_timber',
       'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered',
       'has_superstructure_rc_engineered', 'has_superstructure_other',
       'legal_ownership_status', 'count_families', 'has_secondary_use',
       'has_secondary_use_agriculture', 'has_secondary_use_hotel',
       'has_secondary_use_rental', 'has_secondary_use_institution',
       'has_secondary_use_school', 'has_secondary_use_industry',
       'has_secondary_use_health_post', 'has_secondary_use_gov_office',
       'has_secondary_use_use_police', 'has_secondary_use_other',
       'damage_grade'

- A description of the data already showed some potential outliers, e.g. in the age column, as the max age of 995 is well above the 75th percentile. Even way above the 99th percentile as it will turn out. We might want to create a new feature for this or cap it so the outliers wont play a significant role for this feature.
  
![image](https://github.com/user-attachments/assets/ec208ae4-a15b-4c24-ab37-0ca673cd1b9b)

- The dataset consists of int64 and object datatypes, with no missing values overall (very convenient!)

- For the beginning we dropped the building_id column and left the rest as it was.

## EDA

Plotting is always a good idea, key takeaways:

- Geo_level_ids seem to be random and rather correspond to a certain location, which makes them categorical even though they are represented as numbers.
- Many distributions are heavily right skewed, like age, height, area. We might want to take care of this later with log-transformations, depending on the models we use and how much time is left.
- Some attributes are heavily imbalanced. It makes sense to plot them again the target variable to check if there is a significant difference - if the minority categories correlate significantly with variations in the target, the feature could still be valuable.
- The distribution of the target variable is left skewed, making the target imbalanced, but not too badly. It should be fine to work with it as it is.
  
![newplot](https://github.com/user-attachments/assets/62a07560-cf48-40c1-a19d-d8f2c60e0dec)

- What I love to do is make use of the mutual information package and see what the most relevant information is stored in the features in correspondance to my target variable. This is only a first scoring and without any context whatsoever, so a final model might decide very differently.

- Key takeaways: Geo_level_ids seem to be of importance, many structure related features also seem to play a role, like foundation type, floor_type, roof_type which makes sense if you consider earthquake induced damage in buildings.

![newplot (2)](https://github.com/user-attachments/assets/27afd78c-5f07-4e73-be6f-8936458d02bd)


## Approach

Our approach consisted out of the following steps:

- Transform the Geo_location_ids into categorical features using BaseN-Encoding (I think we used N=5 at the end but experimented with N=3, N=7)
- Unskew skewed numerical columns by using a logarithmic transformation (this only helped marginally but still)
- Target Re-encoding (from 1, 2, 3 -> 0, 1, 2) as xgb classifier requires it.
- Train a baseline classifier (Random Forrest) with arbitrary hyperparameters.
- Use lightGBM and XGBoost as advanced tree-based models.
- Utilize MLFlow for experiment tracking and bayesian hyperparameter search.


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
