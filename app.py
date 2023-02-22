from flask import Flask, render_template, request
import Diabetes_model
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])

def basic():
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin=request.form['Insulin']
        BMI=request.form['BMI']
        DiabetesPedigreeFunction=request.form['DiabetesPedigreeFunction']
        Age=request.form['Age']

        y_pred = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
        trained_model = Diabetes_model.training_model()
        prediction_value = trained_model.predict(y_pred)
        diabetic = 'It seems to be u are diabetic please contact doctor'
        None_diabetic = 'Enjoy! You are fine'
        virginica = 'The flower is classified as Virginica'
        if prediction_value == 0:
            return render_template('index.html',  None_diabetic= None_diabetic)
        elif prediction_value == 1:
            return render_template('index.html',  diabetic= diabetic)
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)