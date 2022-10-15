import joblib

#load model
model=joblib.load("diabetes_79.pkl")

result=model.predict([[1,93,70,87,0,30.4,0.315,23]])

if result[0]==0:
    print("diabetic")
else:
    print("not diabetic")