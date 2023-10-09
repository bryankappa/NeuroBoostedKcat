import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import os

def load_data():
    preprocessor = DataPreprocessor(r'C:\Users\Gilbert\Documents\BCB_Research\NeuroBoostedKcat\data\kcat_transferase.csv')
    return preprocessor

#creating inputs and assigning columns.
class DataPreprocessor:
    def __init__(self, csv_file_path): 
        self.csv_file_path = csv_file_path

    def assign_column(self):
        data = pd.read_csv(self.csv_file_path)
        data.columns = ["EC_number", "Species", "smiles", "Compound_name", "Amino_encoding", "Kcat", "unit"]
        data = pd.DataFrame(data)
        return data
    
    def compute_molecular_weight(self, data):
        def molecular_weight(compound):
            mol = Chem.MolFromSmiles(compound)
            if mol:
                return Descriptors.MolWt(mol)
            else:
                return None

        data["Molecular_Weight"] = data["Compound"].apply(molecular_weight)
        return data

    #applying log to the data
    def apply_log_transform(self, data):
        data["Kcat"] = np.log10(data["Kcat"])
        return data


class PCA:
    def __init__(self, csv_file, n_components=None):
        self.data = pd.read_csv(csv_file)
        self.n_components = n_components
        self.transformed_data = None

    def apply_pca(self):
        # Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(self.data)

        # Apply PCA
        pca = sklearnPCA(n_components=self.n_components)
        self.transformed_data = pca.fit_transform(standardized_data)
        return self.transformed_data

    def set_n_components(self, n_components):
        self.n_components = n_components

    def get_explained_variance_ratio(self):
        if self.transformed_data is None:
            raise ValueError("PCA hasn't been applied yet. Call apply_pca() first.")
        pca = sklearnPCA(n_components=self.n_components)
        pca.fit(self.data)
        return pca.explained_variance_ratio_

    def save_transformed_data(self, output_file):
        if self.transformed_data is None:
            raise ValueError("PCA hasn't been applied yet. Call apply_pca() first.")
        transformed_df = pd.DataFrame(self.transformed_data)
        transformed_df.to_csv(output_file, index=False)

'''
Use Cases for PCA
# pca_instance = PCA('path_to_csv_file.csv', n_components=2)
# pca_instance.apply_pca()
# pca_instance.save_transformed_data('transformed_data.csv') Optional
'''

class AminoEncoder:

    def __init__(self):
        # Define the standard amino acids plus the padding character 'X'
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

    def pad_sequence(self, seq, max_length):
        """Function to pad the sequences."""
        return seq.ljust(max_length, 'X')

    def one_hot_encode(self, seq):
        """One-hot encode the sequences."""
        return [[1 if amino == aa else 0 for amino in self.amino_acids] for aa in seq]

    def encode_csv(self, csv_file):
        """Load the CSV, pad, and one-hot encode the amino sequences."""
        
        data = pd.read_csv(csv_file)
        if 'Amino_encoding' not in data.columns:
            raise ValueError("The CSV does not have the 'Amino_encoding' column")

        # Find the length of the longest sequence
        max_length = data['Amino_encoding'].str.len().max()

        # Apply padding to sequences
        data['padded_sequence'] = data['Amino_encoding'].apply(lambda x: self.pad_sequence(x, max_length))

        # One-hot encode the padded sequences
        encoded_sequences = data['padded_sequence'].apply(self.one_hot_encode)

        # Convert the encoded sequences to a list of lists
        encoded_list = encoded_sequences.apply(lambda x: [item for sublist in x for item in sublist]).tolist()

        # Create a DataFrame from the flattened one-hot encoded list
        encoded_df = pd.DataFrame(encoded_list)

        # Rename columns for clarity
        encoded_df.columns = [f'Pos_{i+1}_{aa}' for i in range(max_length) for aa in self.amino_acids]

        return encoded_df
    
'''
encoder = amino_encoder()
encoded_data = encoder.encode_csv('path_to_your_csv.csv')

'''