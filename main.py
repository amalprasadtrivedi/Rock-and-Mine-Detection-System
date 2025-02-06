# Steps
# 1 Get Data
# 2 Preprocess Data
# 3 Train test split
# 4 Train & evaluate Model 5 Making a Predictive System
# 6 Creating a website

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

solar_data = pd.read_csv('dataset.csv',header=None)
solar_data.head()

solar_data.shape

solar_data.duplicated().sum()


solar_data[60].value_counts()

solar_data.groupby(60).mean()

solar_data.describe()

X = solar_data.drop(columns=60,axis=1)
y = solar_data[60]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify=y, random_state=1)

print(X_train.shape,X_test.shape,'\n')
print(y_train.shape,y_test.shape)

model = LogisticRegression()
model.fit(X_train,y_train)

training_prediction = model.predict(X_train)
print(accuracy_score(training_prediction,y_train))

test_prediction = model.predict(X_test)
print(accuracy_score(test_prediction,y_test))

input_data = (0.0191,0.0173,0.0291,0.0301,0.0463,0.0690,0.0576,0.1103,0.2423,0.3134,0.4786,0.5239,0.4393,0.3440,0.2869,0.3889,0.4420,0.3892,0.4088,0.5006,0.7271,0.9385,1.0000,0.9831,0.9932,0.9161,0.8237,0.6957,0.4536,0.3281,0.2522,0.3964,0.4154,0.3308,0.1445,0.1923,0.3208,0.3367,0.5683,0.5505,0.3231,0.0448,0.3131,0.3387,0.4130,0.3639,0.2069,0.0859,0.0600,0.0267,0.0125,0.0040,0.0136,0.0137,0.0172,0.0132,0.0110,0.0122,0.0114,0.0068)
input_data_np_array = np.asarray(input_data)
reshaped_input = input_data_np_array.reshape(1,-1)
prediction = model.predict(reshaped_input)

if prediction[0] == 'R':
    print('The Object is a Rock')
else:
    print('The Object is a Mine')


