import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Function to create subplots for numerical and categorical features
def create_feature_subplots(df):
    # Separate numerical and categorical features
    numerical_df = df.select_dtypes(include=['number'])
    categorical_df = df.select_dtypes(include=['object', 'category'])
    
    # Calculate the total number of features
    total_features = len(numerical_df.columns) + len(categorical_df.columns)
    print(total_features)
    
    # Calculate the number of rows needed for the subplots (3 columns layout)
    num_rows = int(np.ceil(total_features / 3))
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows, 
        cols=3, 
        subplot_titles=[f"Distribution of {col}" for col in numerical_df.columns] + [f"Distribution of {col}" for col in categorical_df.columns],
        shared_xaxes=False
    )
    
    # Color palette for plots
    colors = px.colors.qualitative.Plotly

    # Plot numerical features as histograms
    for i, col in enumerate(numerical_df.columns):
        # Calculate the correct row and column position
        row = (i // 3) + 1
        col_position = (i % 3) + 1
        
        # Add histogram trace without binning
        fig.add_trace(
            go.Histogram(
                x=numerical_df[col],
                name=col,
                marker=dict(color=colors[i % len(colors)], line=dict(width=1, color='black')),
                opacity=0.75,
                showlegend=True
            ),
            row=row,
            col=col_position
        )
        
    # Plot categorical features as bar plots
    start_index = len(numerical_df.columns)
    for j, col in enumerate(categorical_df.columns):
        # Calculate the correct row and column position
        row = ((start_index + j) // 3) + 1
        col_position = ((start_index + j) % 3) + 1
        
        # Calculate value counts for the categorical feature
        value_counts = categorical_df[col].value_counts()
        
        # Add bar trace
        fig.add_trace(
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=col,
                marker=dict(color=colors[(start_index + j) % len(colors)], line=dict(width=1, color='black')),
                opacity=0.75,
                showlegend=True
            ),
            row=row,
            col=col_position
        )
    
    # Update layout for a better look
    fig.update_layout(
        height=300 * num_rows,
        width=1200,
        title_text="Distributions of Features",
        title_x=0.5,  # Center the title
        template='plotly_white'  # Use a clean template
    )
    
    # Update x-axis and y-axis titles
    fig.update_xaxes(title_text="Value", showgrid=True)
    fig.update_yaxes(title_text="Count", showgrid=True)

    fig.show()
    # Show the figure
    return fig



def get_corr(numerical_df):
    corr_matrix = numerical_df.corr()
    
    fig = plt.figure(figsize=(12, 10))
    
    # Create a heatmap
    map = sns.heatmap(corr_matrix, 
                annot=True,         # Show correlation coefficients
                fmt=".2f",          # Format the annotations to two decimal places
                cmap='coolwarm',    # Color map
                linewidths=0.5,     # Lines between squares
                square=True,        # Make squares
                cbar_kws={"shrink": .5})  # Color bar size
    return fig


def compute_mi_classification(X, y):
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
    mi_scores = mutual_info_classif(X_encoded, y)
    # Create a dictionary with column names and their MI scores
    mi_dict = dict(zip(X.columns, mi_scores))
    mi_dict_sorted = dict(sorted(mi_dict.items(), key=lambda item: item[1], reverse=True))
    #return mi_dict_sorted
    
    # Example usage:
    #mi_scores = compute_mi_classification(x, y)
    
    # Create a bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(mi_dict_sorted.keys()),
            y=list(mi_dict_sorted.values()),
            marker=dict(color='skyblue'),
            text=[f"{v:.3f}" for v in mi_dict_sorted.values()],
            textposition='auto'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Mutual Information Scores with target",
        xaxis_title="Features",
        yaxis_title="Mutual Information Score",
        template='plotly_white',
        height=500
    )
    
    # Show the figure
    fig.show()
    return fig


from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
import plotly.graph_objs as go

def compute_mi_regression(X, y):
    # Encode categorical features
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
    
    # Compute mutual information scores for regression
    mi_scores = mutual_info_regression(X_encoded, y)
    
    # Create a dictionary with column names and their MI scores
    mi_dict = dict(zip(X.columns, mi_scores))
    mi_dict_sorted = dict(sorted(mi_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Create a bar chart using Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=list(mi_dict_sorted.keys()),
            y=list(mi_dict_sorted.values()),
            marker=dict(color='skyblue'),
            text=[f"{v:.3f}" for v in mi_dict_sorted.values()],
            textposition='auto'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Mutual Information Scores with target",
        xaxis_title="Features",
        yaxis_title="Mutual Information Score",
        template='plotly_white',
        height=500
    )
    
    # Show the figure
    fig.show()
    return fig

# Example usage:
# fig = compute_mi_regression(X, y)


def create_scatter_plot(data, feature):
    """
    Creates a scatter plot for a high cardinality numerical feature to show its distribution.

    Parameters:
    data (DataFrame): The data containing the feature.
    feature (str): The name of the high cardinality numerical feature.

    Returns:
    fig (plotly.graph_objs.Figure): The Plotly figure object.
    """
    # Create a scatter plot to show individual points distribution
    fig = go.Figure(data=[
        go.Scatter(
            y=data[feature],
            mode='markers',
            marker=dict(color='skyblue', opacity=0.6),
            text=data.index
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=f"Distribution of {feature} (Scatter Plot)",
        xaxis_title="Index",
        yaxis_title=feature,
        template='plotly_white',
        height=500
    )
    
    # Show the figure
    fig.show()
    return fig
    pass
