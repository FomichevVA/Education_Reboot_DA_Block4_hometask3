import streamlit as st
import pandas as pd
# import sklearn as skl
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv('housing.csv')

if st.button('Отобразить первые пять строк'):
    st.write(df.head())

_TEST_SIZE = st.slider('Укажите долю тестовой выборки', 0.05, 0.40, 0.2, 0.05)

if st.button(f'Обучить модель (test_size: {_TEST_SIZE})'):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=_TEST_SIZE,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))