from flask import Flask, render_template,request
import pandas as pd
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Read the CSV file
df = pd.read_csv("diabetes.csv")
input_data=[]
input_data_as_numpy_array=[]

X_train = df.iloc[:, :8]
Y_train = df.iloc[:, 8] 

# Initialize classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=1)
knn_classifier = KNeighborsClassifier(n_neighbors=20, weights='distance', p=7)

# Fit classifiers to the scaled training data
rf_classifier.fit(X_train, Y_train)
knn_classifier.fit(X_train, Y_train)

# Create a Voting Classifier with soft voting
voting_classifier = VotingClassifier(estimators=[
    ('rf', rf_classifier),
    ('knn', knn_classifier)
], voting='soft')

# Train the Voting Classifier
voting_classifier.fit(X_train, Y_train)






app = Flask(__name__)

@app.route("/")
def main():
    return render_template("into.html")

@app.route("/diabetes")
def diabetes():
    return render_template('diabetes.html')

@app.route("/prediction",methods=['POST'])
def prediction():
    input_data = [x for x in request.form.values()]
    input_data_as_numpy_array = np.asarray(input_data)
    newd = np.array([
    int(input_data_as_numpy_array[4]),
    int(input_data_as_numpy_array[5]),
    int(input_data_as_numpy_array[6]),
    int(input_data_as_numpy_array[7]),
    int(input_data_as_numpy_array[8]),
    float(input_data_as_numpy_array[9]),
    float(input_data_as_numpy_array[10]),
    int(input_data_as_numpy_array[11])])
    input_data_reshaped = newd.reshape(1, -1)
    pred = voting_classifier.predict(input_data_reshaped)
    print(pred)
    if(pred[0]==1):
        toxt="Diabetes Predicted"
        tt="This patient is diagnosed with Diabetes Mellitus based on the provided medical parameters. Further evaluation and management are recommended."
    else:
        toxt="Diabetes Not Predicted"
        tt="This patient is not diagnosed with Diabetes Mellitus based on the provided medical parameters."
    return render_template("predict.html",name=input_data_as_numpy_array[0],age=input_data_as_numpy_array[11],dob=input_data_as_numpy_array[2],bg=input_data_as_numpy_array[3],bmi=input_data_as_numpy_array[9],bp=input_data_as_numpy_array[6],gl=input_data_as_numpy_array[5],il=input_data_as_numpy_array[8],st=input_data_as_numpy_array[7],p=input_data_as_numpy_array[4],tet=toxt,tot=tt)




if __name__ == "__main__":
    app.run(debug=True, port=2500)
