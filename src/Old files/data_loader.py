import pandas as pd

class DataLoader:
    def __init__(self, file_path_values, file_path_labels):
        self.file_path_values = file_path_values
        self.file_path_labels = file_path_labels

    def load_csv(self):
        """Load both values and labels data from CSV files."""
        try:
            values_data = pd.read_csv(self.file_path_values)
            labels_data = pd.read_csv(self.file_path_labels)
            print("Data loaded successfully.")
            return values_data, labels_data
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None, None

    def merge_data(self, values_data, labels_data, on_column):
        """Merge the values and labels data on a common column (e.g., 'id')."""
        try:
            merged_data = pd.merge(values_data, labels_data, on=on_column)
            print("Data merged successfully.")
            return merged_data
        except KeyError as e:
            print(f"Error: {e} - Check if the merge key exists in both dataframes.")
            return None


from data_loader import DataLoader

# Specify file paths
values_file_path = "../data/raw/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_values.csv"
labels_file_path = "../data/raw/Richters_Predictor_Modeling_Earthquake_Damage_-_Train_labels.csv"

# Create an instance of DataLoader
data_loader = DataLoader(values_file_path, labels_file_path)

# Load both values and labels data
values_data, labels_data = data_loader.load_csv()

# Merge the two dataframes on a common column (e.g., 'id')
if values_data is not None and labels_data is not None:
    merged_data = data_loader.merge_data(values_data, labels_data, on_column='building_id')

    # Display merged data
    if merged_data is not None:
        print(merged_data.head())

  