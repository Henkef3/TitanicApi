from flask import Flask, render_template, request, url_for, jsonify

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings 

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

from sklearn.externals import joblib

app = Flask(__name__)


def train(df_ohe):
    dependant_variable = 'Survived'
    
    x = df_ohe[df_ohe.columns.difference([dependant_variable])]

    model_columns = list(x.columns)

    joblib.dump(model_columns,'model_columns.pk1')

    y = df_ohe[dependant_variable]

    lr = LogisticRegression()

    lr.fit(x,y)
    
    joblib.dump(lr,'model.pk1')



def loadDataSet():  

    url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"

    df = pd.read_csv(url)

    df_ = df.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)

   # df_ = df_.drop(['Survived'], axis=1)       # remove the dependent variable from the dataframe X

    df_ohe = transform(df,df_)

    return df_ohe

def transform(df, df_):

    from sklearn.preprocessing import LabelEncoder
    labelEncoder_X = LabelEncoder()
    df_.Sex=labelEncoder_X.fit_transform(df_.Sex)

    # fill the two values with one of the options (S, C or Q)
    row_index = df_.Embarked.isnull()
    df_.loc[row_index,'Embarked']='S' 

    Embarked  = pd.get_dummies(  df_.Embarked , prefix='Embarked'  )
    df_ = df_.drop(['Embarked'], axis=1)
    df_= pd.concat([df_, Embarked], axis=1)  
    # we should drop one of the columns
    df_ = df_.drop(['Embarked_S'], axis=1)

    # Change name
    got= df.Name.str.split(',').str[1]
    df_.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]

    # average age 
    ax = plt.subplot()
    ax.set_ylabel('Average age')
    df_.groupby('Name').mean()['Age'].plot(kind='bar',figsize=(13,8), ax = ax)

    title_mean_age=[]
    title_mean_age.append(list(set(df_.Name)))  #set for unique values of the title, and transform into list
    title_mean_age.append(df_.groupby('Name').Age.mean())
    title_mean_age

    #fill in missing ages
    n_traning= df.shape[0]   #number of rows
    n_titles= len(title_mean_age[1])
    for i in range(0, n_traning):
        if np.isnan(df_.Age[i])==True:
            for j in range(0, n_titles):
                if df_.Name[i] == title_mean_age[0][j]:
                    df_.Age[i] = title_mean_age[1][j]

    df_=df_.drop(['Name'], axis=1)

    return df_

@app.route('/predict',methods=['POST'])

def predict():
    
    
    if lr:

        try:

            json_ = request.get_json()

            query_df = pd.DataFrame(json_)

            query = pd.get_dummies(query_df)

            query = query.reindex(columns=model_columns,fill_value = 0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})
        
        except:

            return jsonify({'trace': "error"})
    else:

        print ("Train the model first")

        return ("No model here to use")

@app.route("/")

def hello():
    return "Welcome to my api"
    
if __name__ == '__main__':

    df_ohe = loadDataSet()

    train(df_ohe)

    lr = joblib.load('model.pk1')
    print('Model loaded')

    model_columns = joblib.load('model_columns.pk1')
    print("Model columns loaded")

    app.run(debug=True, port=12345)
