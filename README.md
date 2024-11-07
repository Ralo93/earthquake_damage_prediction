# Earthquake Damage Prediction
<p align="center"> <img src="https://github.com/user-attachments/assets/f7475789-6abb-4b2f-9ff0-035835c9c0c5" alt="Earthquake Damage"> </p> <p align="center"> <em>Image source: drivendata.org.</em> </p>
This repository contains our team's approach to the Earthquake Damage Prediction competition hosted on DrivenData.org. 

### The challenge and the results

- Submit a solution within two days, while managing a team of three.
- We were able to score the highest micro-F1-score out of all participants in the mini-competition hosted by DataScienceRetreat.
- With less than 1% below rank 1 in the overall competition, this project was a success with just six submissions. Notably, the top-ranking models likely overfitted to the hidden test set.
  
## Project Overview

This project focuses on predicting earthquake-induced damage to buildings based on geospatial and structural data. We used DrivenData’s earthquake dataset, which contains data on approximately 260,601 buildings from 2015.

Our approach consisted of EDA, data cleaning, preprocessing and trying different supervised learning algorithms. Further it included hyperparameter tuning with cross-validation using bayesian search.

As time was a constraint, we worked mainly in jupter labs (I know, not the perfect IDE but for a quick prototyping and EDA etc. actually a good fit in my opinion).


## Table of Contents

- [Data](#data)
- [Approach](#approach)
- [Models](#models)
- [Experiments](#experiments)
- [Results](#results)


## Data
A first glimpse at the data gave us the following information:

- The dataset consists of 260601 rows and 40 columns, one of which is the target, called "damage_grade".

## Key Dataset Columns
Identifiers and Geographic Information

building_id: Unique identifier for each building.
geo_level_1_id, geo_level_2_id, geo_level_3_id: Geographic levels.
Building Characteristics

count_floors_pre_eq: Number of floors before the earthquake.
age: Age of the building.
area_percentage: Floor area as a percentage of the property.
height_percentage: Height as a percentage.
Structural and Material Information

foundation_type: Type of foundation.
roof_type: Type of roof.
ground_floor_type: Type of ground floor.
plan_configuration: Building's structural configuration.
has_superstructure_rc_engineered: Presence of engineered reinforced concrete.
has_superstructure_timber: Presence of timber in the structure.
Secondary Use Indicators

has_secondary_use: Indicates if there is any secondary use.
has_secondary_use_agriculture: Agricultural use.
has_secondary_use_school: School use.
Target Variable

damage_grade: Assigned damage grade for the building.

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

- Having a look at the numerical features, for spotting highly correlated features that might influence statistical tests if a certain model is chosen.
  Removing highly correlated features or creating new features from them can also be a possible way to go.

  ![image](https://github.com/user-attachments/assets/04b5ca73-5db3-4418-8aad-d8c5f01477c4)

- What I love to do is make use of the mutual information package and see what the most relevant information is stored in the features in correspondance to my target variable. This is only a first scoring and without any context whatsoever, so a final model might decide very differently.

- Key takeaways: Geo_level_ids seem to be of importance, many structure related features also seem to play a role, like foundation type, floor_type, roof_type which makes sense if you consider earthquake induced damage in buildings.

![newplot (2)](https://github.com/user-attachments/assets/27afd78c-5f07-4e73-be6f-8936458d02bd)


## Approach

Our approach consisted out of the following steps:

- Extensive EDA and data cleaning
- Preprocessing and encoding of categorical variables
- Hyperparameter tuning using Bayesian search, tracked with MLFlow
- Model experimentation with supervised learning algorithms


## Models

We experimented with several models, including:

1. RandomForrestClassifier (used as a baseline)
3. XGBoostClassifier (known for performing well on tabular data and classification tasks)
4. LightGBMClassifier (also known to perform quite well, sometimes can perform better than xgboost)

Model Selection Considerations

- We prioritized models suitable for tabular data and imbalanced classes, ultimately selecting gradient boosting methods (XGBoost and LightGBM) for their scalability and performance.
- Due to time constraints, we did not try ensemble methods like stacking but recognize them as potential improvements for future work.

## Experiments

Our experimental process included:

- Removing outliers at the 97th percentile
- Transforming skewed data
- Encoding categorical features, like geo_level_ids, using Base-N encoding
- Reducing dimensionality of geoID features through auto-encoding
- Transforming age data to handle outliers
- Hyperparameter tuning using Bayesian search (hyperopt package)

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

Rank on the competition hosted by Drivendata.org:

![image](https://github.com/user-attachments/assets/d692bf77-cc40-4b4f-9a5f-16734166c145)

Snippet of trained models using MLFlow:
![image](https://github.com/user-attachments/assets/0c700040-ab00-47a5-b43c-afc32e38504c)
