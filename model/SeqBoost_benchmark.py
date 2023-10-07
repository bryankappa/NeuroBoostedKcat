import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
import lightgbm as lgb
import xgboost as xgb

class SeqBoost:
    def __init__(self):
        self.model_nn = None
        self.model_gbm = None
        self.meta_model_xgb = None

    def _initialize_nn(self, input_dim):
        model = Sequential()
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_dim,)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _initialize_gbm(self):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
        }
        return params

    def train_base_models(self, X_train, y_train, X_val=None, y_val=None):
        # If validation data isn't provided, split the training data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Train NN
        self.model_nn = self._initialize_nn(X_train.shape[1])
        self.model_nn.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val))

        # Train GBM
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        self.model_gbm = lgb.train(self._initialize_gbm(), lgb_train, num_boost_round=500, valid_sets=[lgb_val])

    def train_meta_model(self, X, y):
        nn_preds = self.model_nn.predict(X)
        gbm_preds = self.model_gbm.predict(X)
        stacked_preds = np.column_stack((nn_preds, gbm_preds))
        
        self.meta_model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.meta_model_xgb.fit(stacked_preds, y)

    def predict(self, X):
        nn_preds = self.model_nn.predict(X)
        gbm_preds = self.model_gbm.predict(X)
        stacked_preds = np.column_stack((nn_preds, gbm_preds))
        return self.meta_model_xgb.predict(stacked_preds)