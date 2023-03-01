from flask import Flask, url_for, request, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nPercentage of fire occuring is {}%'.format(float(output)*100),bhai="danger")
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Percentage of fire occuring is {}%'.format(float(output)*100),bhai="safe")


if __name__ == '__main__':
    app.run(debug=True)