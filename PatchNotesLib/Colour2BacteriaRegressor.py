import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Regressor:
    def train(self, X_train, y_train):
        self.regressor = RandomForestRegressor(n_estimators=100, max_features=3, random_state=0)
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        return self.regressor.predict(X_test)