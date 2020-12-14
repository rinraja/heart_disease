#Supervised learning
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", header=0)
#Print out the top of the table 
#print(data.head(), data.shape)

#Split up the x and y variables because we are going to predict the cost based off of the factors
predictors = data.drop(columns='target')
response = data['target']

#Split the training set from the testing set
#When the test size was increased, the accuracy decreased 
x_train, x_test, y_train, y_test = train_test_split(predictors, response, test_size=0.6)

#Naive Bayes 
model = GaussianNB()
y_pred = model.fit(x_train, y_train).predict(x_test)
print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, y_pred))

#Confusion Matrix. Checking underfitting 
confusion = metrics.confusion_matrix(y_test, y_pred)
#print(confusion)
true_neg = confusion[0][0]
false_neg = confusion[1][0]
true_pos = confusion[1][1]
false_pos = confusion[0][1]
print("type i errs:", false_pos, "type ii errs:", false_neg)
print("Default Accuracy: ", true_pos/(true_pos + false_neg)) 
#The true accuracy is better than the naive bayes value

#Check the training data 
y_pred_train = model.predict(x_train)
print("Training Accuracy:", metrics.accuracy_score(y_train, y_pred_train))
#The training accuracy is 10% higher than the naive bayes 
#There is some overfitting here because the percentage is much higher

#Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))

#Modify the structure of the treebut limit the structure in the tree
clf = DecisionTreeClassifier(min_samples_leaf=10)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy (min leaf 10):", metrics.accuracy_score(y_test, y_pred))
y_pred_train = clf.predict(x_train)
print("Training Accuracy (min leaf 10):", metrics.accuracy_score(y_train, y_pred_train))

clf2 = DecisionTreeClassifier(min_samples_leaf=5)
clf2 = clf2.fit(x_train, y_train)
y_pred2 = clf2.predict(x_test)

print("Accuracy (min leaf 5):", metrics.accuracy_score(y_test, y_pred2))
y_pred_train2 = clf2.predict(x_train)
print("Training Accuracy (min leaf 5):", metrics.accuracy_score(y_train, y_pred_train2))

#Display the model
fig = plt.figure(figsize=(80,40))
_ = tree.plot_tree(clf, feature_names=predictors.columns, filled=True, fontsize=10)
fig.savefig("heartdisease_tree.png")