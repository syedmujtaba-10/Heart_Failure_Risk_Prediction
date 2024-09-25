import urllib.request
import json
import os
import ssl
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')


def allowSelfSignedHttps(allowed):
     # bypass the server certificate verification on client side
     if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
         ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

import openai

@app.route('/predict',methods=['POST'])
def predict():

    data =  {
      "data": [
        {
          "age": request.form['age'],
          "anaemia": request.form['anaemia'],
          "creatinine_phosphokinase": request.form['creatinine_phosphokinase'],
          "diabetes": request.form['diabetes'],
          "ejection_fraction": request.form['ejection_fraction'],
          "high_blood_pressure": request.form['high_blood_pressure'],
          "platelets": request.form['platelets'],
          "serum_creatinine": request.form['serum_creatinine'],
          "serum_sodium": request.form['serum_sodium'],
          "sex": request.form['sex'],
          "smoking": request.form['smoking'],
          "time": request.form['time']
        }
      ],
      "method": "predict"
    }

    body = str.encode(json.dumps(data))

    url = 'http://d637e8a9-cad4-4ce9-b5bc-46ffd5ece113.eastus2.azurecontainer.io/score'
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = '[my api key]'
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")


    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        res=[]
        res=result.split()
        print(res)
        if res[1]== b'[1]}"':
            result1="High Risk"
        elif res[1]== b'[0]}"':
            result1="Low Risk"
        print(result1)

        return render_template('index.html', prediction_text='Prediction: {}'.format(result1))
        #return render_template('predictor.html', answer='Prediction: {}'.format(answer))




    except urllib.error.HTTPError as error:
        return "The request failed with status code: " + str(error.code)

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        # print(error.info())
        # print(error.read().decode("utf8", 'ignore'))

if __name__ == "__main__":
    app.run(debug=True)
