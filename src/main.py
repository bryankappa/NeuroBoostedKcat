import sys
import numpy as np
sys.path.append('C:/Users/Gilbert/Documents/BCB_Research/NeuroBoostedKcat/')
from utilities.data_preprocessor import *
from utilities.performance_metrics import *
from model.SeqBoost_benchmark import *

preprocessor = load_data() # loads main data
data = preprocessor.assign_column() # assings columns to main data
data = preprocessor.apply_log_transform(data)
amino_pca = PCA(r'C:\Users\Gilbert\Documents\BCB_Research\NeuroBoostedKcat\data\encoded_amino copy.csv', n_components=433)
amino_pca = amino_pca.apply_pca()

X = amino_pca
y = data["Kcat"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split training data into training and validation sets
X_train_base, X_val, y_train_base, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Instantiate the SeqBoost class
boosted_model = SeqBoost()

# Train the base models
boosted_model.train_base_models(X_train_base, y_train_base, X_val, y_val)

# Train the meta-model using predictions from the base models on the test set
boosted_model.train_meta_model(X_test, y_test)

# Predict using the stacked model
predictions = boosted_model.predict(X_test)

evaluate_model(y_test, predictions)
