# -*- coding: utf-8 -*-
"""Get Pseudo Labels.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mKsyvxidvBXW4RRc2Y5xT4jkyhbJG6_Y
"""

import numpy as np
import pandas as pd

from google.colab import drive
drive.mount("/content/gdrive")

df = pd.read_csv("/content/gdrive/My Drive/clean_final_embeddings/df1_1.csv")

df.drop('Unnamed: 0', axis=1, inplace=True)

df

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

filename = "/content/gdrive/My Drive/models/LogReg.joblib"
LogReg = joblib.load(filename)

pred = LogReg.predict(df)
pred_fake = pred[np.where(pred == 1)]
pred_real = pred[np.where(pred == 0)]

print(pred.size)
print(pred_fake.size)
print(pred_real.size)

prob = pd.DataFrame(LogReg.predict_proba(df))
prob['max'] = prob.apply(lambda x: max(x[0],x[1]), axis=1)
prob[prob['max'] >= 0.90]