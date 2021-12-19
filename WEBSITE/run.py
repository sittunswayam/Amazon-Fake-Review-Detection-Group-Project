from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from gibberish_classifier import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/manual')
def manual():
    return render_template('manual.html')

def get_clean_embeddings(embeddings):
    data = str(embeddings)
    data = data.replace('[',"")
    data = data.replace(']',"")
    list_of_nums = data.split(" ")

    clean_list = []
    final_clean_list = []

    for entry in list_of_nums:
        if (entry != ''):
            clean_list.append(entry)

    final_clean_list = [[float(entry) for entry in clean_list]]

    return final_clean_list

def get_embeddings(data):
    st_model = SentenceTransformer('./st_model')
    embeddings = st_model.encode(data)
    clean_embeddings = get_clean_embeddings(embeddings)
    
    return clean_embeddings

def get_pd_df(clean_embeddings):
    df = pd.DataFrame(clean_embeddings)
    return df

def extract_from_url(url):
    import requests
    from bs4 import BeautifulSoup
    import random
    import time

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
    ,'Cookie':'x-acbcn=EwNBz6OLTIFDxQCv1qiUE4m16A00AUKs; at-main=Atza|IwEBINcqsHbV-1tFBCYlshzjTAyv5Z4msKVZ0rbOATXYrjE7AcoO3LSnYDzYpZcY2C4WP3oOPIlqWLWh9UcAzDHu6Xv6xcdbCW7jQ59cifSfpYiv3UQ0qR5Hk2VJjX0dcrsdgJUw-TWW8ZWLLhs2Z_CTD7Mphdn9fgvg7qnREuayGRpxekotq9lRXxeqJn3-IfoanhF9edDc0MYk2jTDtJv0AiJp71Wwo6PsNRTwwCg0JS69-H5QYeRbXfFSP-dTtVSGzB-MgVo4zX6dRSmYQ12_rjbfZa7ihj0s-3KtBFLnVP-R91VJrvDwMBSjfcyJHL734UfSrN6D6c1MCq76NoM-MpzmKncsn3n7Ruhnxork43k0onNA0jTl4SD1UDQ8dweuxP6FN0O7eTrWTaBkP_isuiDI; sess-at-main="Wyf/mENo8M2ZhLuc1RWCf++uvPG19jd3RE0X61PIhrk="; x-wl-uid=12Hr4lOV8Md2tj2TjdgVpNVGb5aL6MrEz19aI0yHjr7FY8N3HsTCe29HlZhe4NCBbeDw2KuN5ShkJajzdy70eGSYuSAIda2OF1CcLpnHo+Bd7mvKvVqTsj1pNwri9d8E2lMOUplbiuZ8=; session-id-time-cn=1482739200l; session-id-cn=452-5760864-5873122; session-token=h1J7fMqt9UYrlp3EVScY8zWkFsNT7oGwBzJLHkKb8ChGVAMO/6quZxt9R24wwGPUCc4BPFLofrOQ5ZG9Jf9KQ5Y7j6XhKqlUh9j3g60qdVTgNSM6gY+eERRbI7iWTLGXQwEBB9LOx49+htkQIMfw1coTjYn50RlfUeeuW9dE8Db937LkwRFJe1ewcyebJZ713u/9HGAFQvCwatOslgNVHrpWOPGW91OUqhkYdW9wS6G46ScDqefXu2tRqWL8mOKn7t4wdMGqaF8=; csm-hit=YWXGRBPABQJZHSN0TYMN+s-88A1TVR7SK2BG3FDDW2M|1482139543670; ubid-acbcn=453-1347853-1253656; session-id-time=2082729601l; session-id=452-5760864-5873122'
    ,'Host':'www.amazon.com'
    ,'X-Requested-With':'XMLHttpRequest'}
    time.sleep(0.5 * random.random())

    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')

    rating = soup.find('span', class_="a-icon-alt").text.split('.')[0]

    verified_purchase = ''
    get_verified_purchase = soup.find('span', class_="a-size-mini a-color-state a-text-bold")
    if (get_verified_purchase):
        verified_purchase = 'Y'
    else:
        verified_purchase = 'N'

    text = soup.find('title').text
    text += ". "
    text += soup.find('span', class_="a-size-base review-text review-text-content").text.replace('\n','').replace('\\','')

    return rating, verified_purchase, text

@app.route('/predict', methods=['POST'])
def predict():
    clf_open = open('LogReg.pkl','rb')
    LogReg = joblib.load(clf_open)

    if request.method == 'POST':
        url = request.form['reviewurl']
        rating, verified_purchase, text = extract_from_url(url)

        new_text = ''
        new_text += "The rating is {}. ".format(rating)
        if (verified_purchase == 'Y'):
            new_text += "It is a verified purchase. "
        if (verified_purchase == 'N'):
            new_text += "It is not a verified purchase. "
        new_text += text

        gibberish = round(gibberishclassifier(text), 2)
        
        if (gibberish <= 45):
            clean_embeddings = get_embeddings(new_text)
            df_to_pred = get_pd_df(clean_embeddings)
            final_prediction = LogReg.predict(df_to_pred)

            prob_df = pd.DataFrame(LogReg.predict_proba(df_to_pred))
            prob_df['max'] = prob_df.apply(lambda x: max(x[0],x[1]), axis=1)
            pred_confidence = round(prob_df['max'][0]*100, 2)
        else:
            final_prediction = -1
            pred_confidence = 100
        
    return render_template('result.html', prediction = final_prediction, confidence = pred_confidence)


@app.route('/predictmanual', methods=['POST'])
def predictmanual():
    clf_open = open('LogReg.pkl','rb')
    LogReg = joblib.load(clf_open)

    if request.method == 'POST':
        title = request.form['reviewtitle']
        text = request.form['message']
        rating = request.form['Rating']
        verified_purchase = request.form['Verified-Purchase']
        product_category = request.form['Product-Category']

        new_text = ''
        if (rating != 'None'):
            new_text += "The rating is {}. ".format(rating)
        if (product_category != 'None'):
            if (verified_purchase == 'Y'):
                new_text += "It is a verified purchase with product category {}. ".format(product_category)
            if (verified_purchase == 'N'):
                new_text += "It is not a verified purchase with product category {}. ".format(product_category)
            if (verified_purchase == 'None'):
                new_text += "The product category is {}. ".format(product_category)
        else:
            if (verified_purchase == 'Y'):
                new_text += "It is a verified purchase. "
            if (verified_purchase == 'N'):
                new_text += "It is not a verified purchase. "
        new_text += title
        new_text += ". "
        new_text += text

        gibberish = round(gibberishclassifier(text), 2)

        if (gibberish <= 45):
            clean_embeddings = get_embeddings(new_text)
            df_to_pred = get_pd_df(clean_embeddings)
            final_prediction = LogReg.predict(df_to_pred)

            prob_df = pd.DataFrame(LogReg.predict_proba(df_to_pred))
            prob_df['max'] = prob_df.apply(lambda x: max(x[0],x[1]), axis=1)
            pred_confidence = round(prob_df['max'][0]*100, 2)
        else:
            final_prediction = -1
            pred_confidence = 100
        
    return render_template('result.html', prediction = final_prediction, confidence = pred_confidence)

if __name__ == '__main__':
    app.run(debug=True)