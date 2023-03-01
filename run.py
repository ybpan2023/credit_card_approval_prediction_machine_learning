from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import datetime
import pandas as pd


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    result = " "
    if request.method == "POST":
        birthday = request.form["birthday"]
        gender = request.form["gender"]
        own_car = request.form["own_car"]
        own_realty = request.form["own_realty"]
        income = request.form["income"]
        incometp = request.form["incometp"]
        edutp = request.form["edutp"]
        famtp = request.form["famtp"]
        housetp = request.form["housetp"]
        occupationtp = request.form["occupationtp"]
        DAYS_EMPLOYED = request.form["DAYS_EMPLOYED"]
        phone = request.form["phone"]
        workphone = request.form["workphone"]
        email = request.form["email"]
        famnum = request.form["famnum"]
        birthday = pd.to_datetime(birthday)
        DAYS_BIRTH = datetime.date.today()- datetime.date(birthday.year, birthday.month, birthday.day)
        DAYS_EMPLOYED = pd.to_datetime(DAYS_EMPLOYED)
        DAYS_EMPLOYED = datetime.date.today()- datetime.date(DAYS_EMPLOYED.year, DAYS_EMPLOYED.month, DAYS_EMPLOYED.day)
        X = np.array([[gender, own_car, own_realty, int(income), incometp, edutp, famtp, housetp,
                       int(DAYS_BIRTH.days), int(DAYS_EMPLOYED.days), workphone, phone, email, occupationtp, int(famnum)]])
        X = pd.DataFrame(X, columns = ['gender', 'own_car', 'own_realty', 'income', 'incometp', 'edutp',
                                       'famtp', 'housetp', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'workphone', 'phone',
                                       'email', 'occupationtp', 'famnum'])
        result = model.predict(X)
        if result == 0:
            result = 'Congradulations! Your credit card application will likely be approved!'
        else:
            result = 'Sorry! We cannot make a decision based on your current information!'
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
