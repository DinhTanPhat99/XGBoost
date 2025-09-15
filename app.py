from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
import pickle

app = Flask(__name__)

# Giả sử bạn đã train model và lưu thành model.pkl
# Nếu chưa có, bạn cần train XGBoost và dump model:
# pickle.dump(model, open("model.pkl", "wb"))

try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    model = None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        income = float(request.form["MonthlyIncome"])
        debt = float(request.form["DebtRatio"])
        open_lines = int(request.form["NumberOfOpenCreditLinesAndLoans"])
        dependents = int(request.form["NumberOfDependents"])

        features = np.array([[age, income, debt, open_lines, dependents]])

        if model:
            pred = model.predict(features)[0]
        else:
            pred = np.random.choice([0, 1])  # fallback random
        return render_template("index.html", prediction=int(pred))
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
