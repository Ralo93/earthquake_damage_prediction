import pandas as pd
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, base_columns, oh_columns):
    """
    Preprocesses the DataFrame by one-hot encoding and base encoding all categorical values,
    and scaling all numeric values.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be preprocessed.
    base_columns (list): List of columns to be base encoded.
    oh_columns (list): List of columns to be one-hot encoded.

    Returns:
    Pipeline: The preprocessing pipeline.
    """
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Define the transformers for categorical and numeric data
    categorical_transformer = Pipeline(steps=[
        ('onehot', ce.OneHotEncoder(cols=oh_columns)),
        ('base', ce.BaseNEncoder(cols=base_columns, base=2))
    ])

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, base_columns + oh_columns),
            ('num', numeric_transformer, numeric_cols)
        ]
    )

    # Create and fit the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    return pipeline