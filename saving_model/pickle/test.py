from cmath import pi
import pickle

#load model
model=pickle.load(open("diabetes_79.pkl",'rb'))

result=model.predict([[1,93,70,87,0,30.4,0.315,23]])

if result[0]==0:
    print("diabetic")
else:
    print("not diabetic")