import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names=['preg','plas','pres','skin','test','mass','predi','age','class']
dataset=pd.read_csv(url,names=names)
print(dataset)

array = dataset.values
X = array[:,0:8]
y = array[:,8]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=101)

# fir the model
model = LogisticRegression()
model.fit(X_train,y_train)

# accuracy
result = model.score(X_test, y_test)
print( 'accuracy : ',result)


# saving the model
filename= "diabetes_79.pkl"
pickle.dump(model, open(filename,'wb'))