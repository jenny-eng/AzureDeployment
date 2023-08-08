import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import math 

app = Flask(__name__)
model = pickle.load(open('aichi_model.pkl','rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    #int_features = [float(x) for x in request.form.values()]
    int_features = [int(x) for x in request.form.values()]
    #int_features = request.form.values().split(",")
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    prediction = np.sum(int_features)
    #print(math.floor(prediction[0]))

    #output = round(prediction[0], 2)
    #return render_template('home.html', prediction_text="Predictd Truck Number is {}".format(math.floor(prediction[0])))
    return render_template('home.html', prediction_text="Predictd Truck Number is {}".format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
