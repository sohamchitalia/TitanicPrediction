import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
 

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_df = train
test_df = test
combined = [train_df, test_df]

# view columns

print(train_df.columns.values)
print(test_df.columns.values)

#view data
train_df.head()

train_df.info()
print('*'*50)
train_df.head()
test_df.info()

#get stats about data

train_df.describe()

#classwise survival
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

#sexwise survival
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

#siblingwise survival
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())

#parent-childen wise survivale

print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())

#age v/s survived graph
graph = sns.FacetGrid(train_df, col='Survived')
graph.map(plt.hist, 'Age', bins=20)
plt.show()


# #survival based on Pclass and age. 
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()


# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# #dropping unnecessary features - ticket and cabin

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

#extracting title from name

for d in combine:
	d['Title'] = d.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)
print(pd.crosstab(train_df['Title'], train_df['Sex']))

for d in combine:
	d['Title'] = d['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	d['Title'] = d['Title'].replace('Mlle', 'Miss')
	d['Title'] = d['Title'].replace('Ms', 'Miss')
	d['Title'] = d['Title'].replace('Mme', 'Mrs')
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for d in combine:
	d['Title'] = d['Title'].map(title_mapping)
	d['Title'] = d['Title'].fillna(0)
print(train_df.head())

#dropping name and passenger id

train_df = train_df.drop(['Name','PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
for d in combine:
	d['Sex'] = d['Sex'].map({'female': 1, 'male': 0}).astype(int)
print(train_df.head())


grid = sns.FacetGrid(train_df, row = 'Pclass', col = 'Sex', size = 2.2, aspect = 1.6)
grid.map(plt.hist, 'Age', alpha = .5, bins=20)
grid.add_legend()
plt.show()


guess_ages = np.zeros((2,3))
print(guess_ages)
for d in combine:
	for i in range(0,2):
		for j in range(0,3):
			guess_df = d[(d['Sex'] == i) & (d['Pclass'] == j+1)]['Age'].dropna()
			age_guess = guess_df.median()
			guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5

	for i in range(0,2):
		for j in range(0,3):
			d.loc[(d.Age.isnull()) & (d.Sex == i) & (d.Pclass == j+1), 'Age'] = guess_ages[i,j]
	d['Age'] = d['Age'].astype(int)
print(train_df.head())

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean() )


for d in combine:
	d.loc[d['Age'] <= 16, 'Age'] = 0
	d.loc[(d['Age'] > 16) & (d['Age'] <= 32), 'Age'] = 1
	d.loc[(d['Age'] > 32) & (d['Age'] <= 48), 'Age'] = 2
	d.loc[(d['Age'] > 48) & (d['Age'] <= 64), 'Age'] = 3
	d.loc[(d['Age'] > 64), 'Age'] = 4


train_df = train_df.drop(['AgeBand'], axis = 1)

combine = [train_df, test_df]

#family siz feautre by combining parch and sibsp

for d in combine:
	d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by = 'Survived', ascending = False))

#isAlone

for d in combine:
	d['IsAlone'] = 0
	d.loc[d['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean())


#drop parch and sibsp

train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

#check for removing family size above

#combining Pclass and age

for d in combine:
	d['Age*Class'] = d['Age'] * d['Pclass']
print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

portfreq = train_df['Embarked'].dropna().mode()[0]
print(portfreq)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(portfreq)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for d in combine:
	d['Embarked'] = d['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(int)


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index = False).mean().sort_values(by='FareBand', ascending = True))


for d in combine:
	d.loc[d['Fare'] <= 7.91, 'Fare'] = 0
	d.loc[(d['Fare'] > 7.91) & (d['Fare'] <= 14.454), 'Fare'] = 1
	d.loc[(d['Fare'] > 14.454) & (d['Fare'] <= 31), 'Fare'] = 2
	d.loc[(d['Fare'] > 31), 'Fare'] = 3
	
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head())
print(test_df.head())


#train test data

X_train = train_df.drop(['Survived'], axis = 1)
Y_train = train_df['Survived']
X_test = test_df.drop(['PassengerId'], axis=1).copy()


#models to predict

#logistic regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print "Accuracy of logistic regression: ", acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

print(coeff_df.sort_values(by='Correlation', ascending = False))

#SVM

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print "Accuracy of SVM ", acc_svc


#KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)
print "Accuracy of KNN ", acc_knn



#Gaussian NB

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print "Accuracy of gaussian NB ", acc_gaussian 


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print "Accuracy of perceptron" ,acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print "Accuracy of linear svc", acc_linear_svc


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print "Accuracy of SGD:", acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print "Accuracy of decision tree: ", acc_decision_tree


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print "Accuracy of random forest: ", acc_random_forest




models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))

answers = pd.DataFrame({
	"PassengerId": test_df["PassengerId"],
	"Survived": Y_pred
	})


#prediction
print(answers)








	

	








