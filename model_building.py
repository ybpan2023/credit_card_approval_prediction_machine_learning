#!/usr/bin/env python
# coding: utf-8

# In[57]:

import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('database.csv')
features = list(df.columns)
target = 'label'
features.remove(target)
X = df[features]
Y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
oe_cols = ['gender', 'own_car', 'own_realty', 'incometp', 
           'edutp', 'famtp', 'housetp', 'workphone', 'phone', 'email', 'occupationtp']
#num_cols = ['income', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'famnum']
#cols = ColumnTransformer([('ordinal', OrdinalEncoder(), oe_cols)])
cols = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown = "ignore"), oe_cols)], remainder="passthrough")
steps = [('cols', cols), ('smt', SMOTE()), ('rescale', StandardScaler(with_mean = False)), 
         ('xgb', XGBClassifier(objective='binary:logistic', 
                                 learning_rate=0.5, max_depth=8,
                                 n_estimators=250, min_child_weight=4))]

model = Pipeline(steps)
model = model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
