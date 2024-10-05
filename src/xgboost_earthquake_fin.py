from sklearn.model_selection import StratifiedKFold
import mlflow
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import BaseNEncoder
import pandas as pd
from xgboost import XGBClassifier
pd.set_option('future.no_silent_downcasting', True)


# Define the search space for hyperparameter optimization
space = {
    "n_estimators": hp.uniformint("n_estimators", 300, 1000),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
    "max_depth": hp.uniformint("max_depth", 3, 7),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1),
    "subsample": hp.uniform("subsample", 0.6, 0.95),
    "min_child_weight": hp.uniformint("min_child_weight", 1, 10),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10)),  # L2 regularization
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(10)),    # L1 regularization
    "min_split_gain": hp.loguniform("min_split_gain", np.log(1e-8), np.log(10))  # Equivalent to gamma
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading Data...")
data = pd.read_csv(r'../data/all_train_data.csv')
logging.info("Done")

pct = np.percentile(data.loc[:, 'area_percentage'].fillna(np.mean(data.loc[:, 'area_percentage'])), 97)
print(pct)
print(data.shape)
data = data.loc[data.loc[:, 'area_percentage'] < pct]
print(data.shape)

pct = np.percentile(data.loc[:, 'height_percentage'].fillna(np.mean(data.loc[:, 'height_percentage'])), 97)
print(pct)
print(data.shape)
data = data.loc[data.loc[:, 'height_percentage'] < pct]
print(data.shape)

class GeoInteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create geo interaction terms by concatenating the geo-level IDs.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        # Concatenate geo_level_1_id, geo_level_2_id, and geo_level_3_id
        X_new['geo1_geo2'] = X_new['geo_level_1_id'].astype(str) + '_' + X_new['geo_level_2_id'].astype(str)
        X_new['geo1_geo3'] = X_new['geo_level_1_id'].astype(str) + '_' + X_new['geo_level_3_id'].astype(str)
        X_new['geo2_geo3'] = X_new['geo_level_2_id'].astype(str) + '_' + X_new['geo_level_3_id'].astype(str)
        X_new['geo_all'] = (
                X_new['geo_level_1_id'].astype(str) + '_' +
                X_new['geo_level_2_id'].astype(str) + '_' +
                X_new['geo_level_3_id'].astype(str)
        )
        # Return the entire dataframe including original and new columns
        return X_new




def get_right_skewed_columns(df, skew_threshold=0.5):
    """
    Returns the names of columns that are right-skewed based on the skewness value, excluding binary columns.

    Parameters:
    - df: The input DataFrame (numerical columns only).
    - skew_threshold: The skewness threshold above which a column is considered right-skewed (default is 0.5).

    Returns:
    - List of column names that are right-skewed.
    """
    right_skewed_columns = []

    # Iterate through each column in the dataframe
    for col in df.columns:
        # Check if the column has more than 2 unique values (to avoid binary columns)
        if df[col].nunique() > 2:
            # Calculate skewness for each column
            col_skewness = skew(df[col].dropna())  # Drop NaN values to avoid issues

            # Check if the skewness is above the specified threshold (indicating right-skewness)
            if col_skewness > skew_threshold:
                right_skewed_columns.append(col)

    return right_skewed_columns

numerical_df = data.select_dtypes(exclude=['object'])

# # Select numerical columns
# numerical_df = data.select_dtypes(exclude=['object'])

# # Get the right-skewed columns
right_skewed_cols = get_right_skewed_columns(numerical_df)

print("Right-skewed columns:", right_skewed_cols)

def log_transform(X):
    # Apply log1p transformation (log(1 + x)) to avoid issues with zero values
    return np.log1p(X)

# Create a FunctionTransformer for log transformation
log_transformer = FunctionTransformer(log_transform)




# Custom transformer for the age-based transformation
class AgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, age_column='age'):
        self.age_column = age_column
        self.percentile_ = None

    def fit(self, X, y=None):
        # Calculate the 99th percentile of the 'age' column and store it
        self.percentile_ = np.percentile(X[self.age_column].fillna(np.mean(X[self.age_column])), 99)
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Add a new 'old' column to indicate if the age exceeds the 99th percentile
        X_copy['old'] = np.where(X_copy[self.age_column] >= self.percentile_, 1, 0)

        # Cap the age to 100 where the 'old' column is 1
        X_copy.loc[X_copy['old'] == 1, self.age_column] = 100

        return X_copy


x = data.drop(columns=['damage_grade'])
y = data.damage_grade
y = y.replace({1: 0, 2: 1, 3: 2})


# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in x.columns if  x[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in x.columns if x[cname].dtype in ['int32', 'int64', 'float64']]

#create numerical transformer
geo_interaction_transformer = GeoInteractionTransformer()

geo_cols = ['geo1_geo2', 'geo1_geo3', 'geo2_geo3', 'geo_all']

# BaseNEncoder for the high cardinality of interaction features
base_encoder_geo = BaseNEncoder(cols=geo_cols, base=5)

# Geo interaction pipeline: First, create interaction terms, then encode them
geo_interaction_pipeline = Pipeline(steps=[
    ('geo_interaction', geo_interaction_transformer),  # Create interaction terms
    ('base_encoder', base_encoder_geo)  # Encode interaction terms
])


numerical_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean'))])

#create categorical transformer
#categorical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='most_frequent')),
#                                            ('onehot', OneHotEncoder(handle_unknown='ignore'))
#                                            ])

base_encoder_columns = ['land_surface_condition','geo_level_1_id', 'geo_level_2_id','geo_level_3_id',
 'foundation_type',
 'roof_type',
 'ground_floor_type',
 'other_floor_type',
 'position',
 'plan_configuration',
 'legal_ownership_status']

base_encoder = Pipeline(steps=[
    ('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))
])

age_transformer = Pipeline(steps=[
    ('age_transform', AgeTransformer(age_column='age'))  # Apply age transformation
])

#height_cluster_transformer = KMeansHeightClusterTransformer(n_clusters=3, column='area_percentage', new_feature_name='height_cluster')

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd


# Define the autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder (optional, only necessary if you want reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# Custom transformer to handle autoencoder training and dimensionality reduction
class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, latent_dim=10, epochs=100, batch_size=32, learning_rate=1e-3):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.autoencoder = None  # Autoencoder model will be initialized in `fit`

    def fit(self, X, y=None):
        # Automatically determine input_dim based on X's shape after BaseN encoding
        input_dim = X.shape[1]  # Number of features in X after BaseN encoding
        self.autoencoder = Autoencoder(input_dim=input_dim, latent_dim=self.latent_dim)

        # Convert X to NumPy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert NumPy array to tensor
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            encoded, decoded = self.autoencoder(X_tensor)
            loss = criterion(decoded, X_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:  # Print loss every 10 epochs for monitoring
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        return self

    def transform(self, X):
        # Convert X to NumPy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.autoencoder.eval()  # Set to evaluation mode
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert NumPy array to tensor
        with torch.no_grad():
            encoded, _ = self.autoencoder(X_tensor)
        return encoded.numpy()  # Return the reduced-dimensional representation


reduction_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

base_encoder_columns = ['land_surface_condition',
                        'foundation_type',
                        'roof_type',
                        'ground_floor_type',
                        'other_floor_type',
                        'position',
                        'plan_configuration',
                        'legal_ownership_status']

base_encoder = Pipeline(steps=[
    ('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))
])

base_encoder_geo = Pipeline(steps=[
    ('base_encoder', BaseNEncoder(cols=reduction_columns, base=3))
])

# Define the pipeline with an Autoencoder for dimensionality reduction
preprocessor = ColumnTransformer(transformers=[

    ('base_name_geo', base_encoder_geo, reduction_columns),  # BaseN encoding on categorical columns

    # Autoencoder dimensionality reduction for BaseN encoded features
    ('autoencoder', Pipeline(steps=[
        ('extractor', FunctionTransformer(lambda x: x, feature_names_out='one-to-one')),
        # Pass through BaseN encoded features
        ('autoencoder', AutoencoderTransformer(latent_dim=5))  # Autoencoder with latent dimension 10
    ]), reduction_columns),
    # ('autoencoder', Pipeline(steps=[
    #     ('extractor', FunctionTransformer(lambda x: x, feature_names_out='one-to-one')),  # Extract original geo columns
    #     ('autoencoder', AutoencoderTransformer(latent_dim=5))  # Autoencoder for geo features with latent dimension 5
    # ]), reduction_columns),

    ('base_name', base_encoder, base_encoder_columns),  # BaseN encoding on categorical columns
    ('age_transform', age_transformer, ['age']),  # Custom transformer for 'age'
    ('num', 'passthrough', numerical_cols),  # Pass through numerical columns without transformation
    ('log_transform', log_transformer, right_skewed_cols)  # Log transform for right-skewed columns
    # Add other transformers or steps as needed
])

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

base_encoder = Pipeline(steps=[('base_encoder', BaseNEncoder(cols=base_encoder_columns, base=3))])


# Consider doing PCA!

def create_model(params):
    XGBClassifier(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        colsample_bytree=params['colsample_bytree'],
        subsample=params['subsample'],
        min_child_weight=int(params['min_child_weight']),
        lambda_l2=params['reg_lambda'],  # L2 regularization
        lambda_l1=params['reg_alpha'],  # L1 regularization
        min_split_gain=params['min_split_gain'],  # Equivalent to gamma
        random_state=42,
        objective='multiclass',
        metric='multi_logloss',
        num_class=3
    )


# Model pipeline
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.2669112505018992,
    max_depth=5,
    random_state=42,
    lambda_l2=1.2259716591605452,  # L2 regularization
    subsample=0.704976942819638,    # Subsample ratio of the training instances
    colsample_bytree=0.9,           # Subsample ratio of columns when constructing each tree
    min_child_weight=4,             # Equivalent of min_data_in_leaf in LightGBM
    lambda_l1=0.14170716330946964,  # L1 regularization term
    objective='multiclass',         # Objective for multiclass classification
    metric='multi_logloss',         # Metric used for multiclass classification
    num_class=3                     # Specify the number of classes in the target
)


rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('xgb', xgb)])

# Upsample only the training data within each fold
def cross_validate_with_downsampling(params, X, y):
    mlflow.xgboost.autolog()

    logging.info("Starting Cross-Validation with Upsampling...")

    rf_pipe.set_params(
    xgb__n_estimators=params['n_estimators'],
    xgb__learning_rate=params['learning_rate'],
    xgb__max_depth=params['max_depth'],
    xgb__subsample=params['subsample'],
    xgb__lambda_l2=params['reg_lambda'],  # L2 regularization
    xgb__colsample_bytree=params['colsample_bytree'],
    xgb__min_child_weight=int(params['min_child_weight']),
    xgb__lambda_l1=params['reg_alpha']  # L1 regularization
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    train_accuracy_scores, train_precision_scores, train_recall_scores, train_f1_scores = [], [], [], []

    use_cross_validation = True  # Set to False to disable cross-validation

    with mlflow.start_run(nested=True):
        if use_cross_validation:
            for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
                logging.info(f"Starting fold {fold_idx + 1}...")

                # Split the data
                X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
                y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

                print('Validation X_shape, y_counts:')
                print(X_valid_fold.shape)
                print(y_valid_fold.value_counts())

                # Train the model on the training data for this fold
                rf_pipe.fit(X_train_fold, y_train_fold)

                # Calculate training metrics
                train_preds = rf_pipe.predict(X_train_fold)
                train_accuracy = accuracy_score(y_train_fold, train_preds)
                train_precision = precision_score(y_train_fold, train_preds, average='micro')
                train_recall = recall_score(y_train_fold, train_preds, average='micro')
                train_f1 = f1_score(y_train_fold, train_preds, average='micro')

                # Calculate validation metrics
                valid_preds = rf_pipe.predict(X_valid_fold)
                fold_accuracy = accuracy_score(y_valid_fold, valid_preds)
                fold_precision = precision_score(y_valid_fold, valid_preds, average='micro')
                fold_recall = recall_score(y_valid_fold, valid_preds, average='micro')
                fold_f1 = f1_score(y_valid_fold, valid_preds, average='micro')

                # Append metrics
                train_accuracy_scores.append(train_accuracy)
                train_precision_scores.append(train_precision)
                train_recall_scores.append(train_recall)
                train_f1_scores.append(train_f1)

                accuracy_scores.append(fold_accuracy)
                precision_scores.append(fold_precision)
                recall_scores.append(fold_recall)
                f1_scores.append(fold_f1)

            # Log cross-validation metrics
            mlflow.log_metric("train_cv_mean_accuracy", np.mean(train_accuracy_scores))
            mlflow.log_metric("train_cv_std_accuracy", np.std(train_accuracy_scores))
            mlflow.log_metric("train_cv_mean_precision", np.mean(train_precision_scores))
            mlflow.log_metric("train_cv_std_precision", np.std(train_precision_scores))
            mlflow.log_metric("train_cv_mean_recall", np.mean(train_recall_scores))
            mlflow.log_metric("train_cv_std_recall", np.std(train_recall_scores))
            mlflow.log_metric("train_cv_mean_f1_score", np.mean(train_f1_scores))
            mlflow.log_metric("train_cv_std_f1_score", np.std(train_f1_scores))

            mlflow.log_metric("cv_mean_accuracy", np.mean(accuracy_scores))
            mlflow.log_metric("cv_std_accuracy", np.std(accuracy_scores))
            mlflow.log_metric("cv_mean_precision", np.mean(precision_scores))
            mlflow.log_metric("cv_std_precision", np.std(precision_scores))
            mlflow.log_metric("cv_mean_recall", np.mean(recall_scores))
            mlflow.log_metric("cv_std_recall", np.std(recall_scores))
            mlflow.log_metric("cv_mean_f1_score", np.mean(f1_scores))
            mlflow.log_metric("cv_std_f1_score", np.std(f1_scores))

            result_loss = -np.mean(f1_scores)

        else:
            logging.info("Training without cross-validation...")

            # Train the model on the full dataset
            rf_pipe.fit(X_train, y_train)

            # Calculate training metrics
            train_preds = rf_pipe.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_preds)
            train_precision = precision_score(y_train, train_preds, average='micro')
            train_recall = recall_score(y_train, train_preds, average='micro')
            train_f1 = f1_score(y_train, train_preds, average='micro')

            # Log training metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1_score", train_f1)
            result_loss = -train_f1

        mlflow.set_tag("tag", "metrics now with opt")

        logging.info("Training process completed")

        return {"loss": result_loss, "status": STATUS_OK, "model": rf_pipe}


def objective(params):
    return cross_validate_with_downsampling(params, x, y)


# Set the MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("/xgb Earthquake regularization")

# Start a new MLflow run for hyperparameter optimization
with mlflow.start_run():
    logging.info("Starting MLflow with Cross-Validation...")

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials
    )

    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    mlflow.log_params(best)
    mlflow.log_metric("micro_f1_score", -best_run["loss"])

    lightgbm = best_run["model"].named_steps['xgb']
    mlflow.sklearn.log_model(rf_pipe, "model")

    print(f"Best parameters: {best}")
    print(f"Best eval macro F1 score: {-best_run['loss']}")
