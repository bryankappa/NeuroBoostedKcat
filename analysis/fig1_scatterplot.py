import sys
import numpy as np
import pandas as pd
sys.path.append('C:/Users/Gilbert/Documents/BCB_Research/NeuroBoostedKcat/')
from utilities.data_preprocessor import *
from utilities.performance_metrics import *
from model.SeqBoost_benchmark import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

preprocessor = load_data() # loads main data
data = preprocessor.assign_column() # assings columns to main data
data = preprocessor.apply_log_transform(data)
amino_pca = PCA(r'C:\Users\Gilbert\Documents\BCB_Research\NeuroBoostedKcat\data\encoded_amino copy.csv', n_components=433)
amino_pca = amino_pca.apply_pca()

X = amino_pca
y = data["Kcat"]

class SeqBoostPipeline:
    def __init__(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.boosted_model = None

    def split_data(self):
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

        # Split training data into training and validation sets
        X_train_base, X_val, y_train_base, y_val = train_test_split(
            X_train, y_train, test_size=self.val_size, random_state=self.random_state
        )

        return X_train_base, X_val, y_train_base, y_val, X_test, y_test

    def train_models(self):
        X_train_base, X_val, y_train_base, y_val, X_test, y_test = self.split_data()

        # Instantiate the SeqBoost class (assuming it's already imported)
        self.boosted_model = SeqBoost()

        # Train the base models
        self.boosted_model.train_base_models(X_train_base, y_train_base, X_val, y_val)

        # Train the meta-model using predictions from the base models on the test set
        self.boosted_model.train_meta_model(X_test, y_test)

    def evaluate(self):
        _, _, _, _, X_test, y_test = self.split_data()

        # Predict using the stacked model
        predictions = self.boosted_model.predict(X_test)

        return predictions, y_test # assuming evaluate_model is already defined

if __name__ == "__main__":
    # Assuming X, y are your feature matrix and labels
    pipeline = SeqBoostPipeline(X, y)
    pipeline.train_models()
    predictions, y_test = pipeline.evaluate()

# Seaborn styling
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 10))
sns.regplot(x=y_test, y=predictions, scatter_kws={'alpha':0.5, 's':60}, line_kws={"color":"red", "linewidth":2}, ci=None)
r2 = r2_score(y_test, predictions)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--')
plt.xlabel("Actual", fontsize=15)
plt.ylabel("Predicted", fontsize=15)
plt.title(f"Actual vs Predicted (R² = {r2:.2f})", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
