##from google.colab import drive
##drive.mount('/content/drive')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# reading the dataset

data = pd.read_csv('online_shoppers_intention.csv')

# checking the shape of the data
data.shape

# checking the head of the data

data.head()

# describing the data

data.describe()

# checking the datatypes of the data

data.dtypes

# taking out the information from the data

data.info()

# checking if the data contains any NULL values

data.isnull().sum()

# checing the distribution of revenue
"""
data['Revenue'].value_counts()

# checking the Distribution of customers on Revenue

plt.rcParams['figure.figsize'] = (10, 5)
sns.countplot(data['Weekend'], palette = 'pastel')
plt.title('Customers who add Revenue to the Company', fontsize = 30)
plt.xlabel('Revenue or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show()


#  checking the distribution of Weekend

data['Weekend'].value_counts()

# checking the Distribution of customers on Weekend

plt.rcParams['figure.figsize'] = (10, 5)
sns.countplot(data['Weekend'], palette = 'colorblind')
plt.title('Distribution of Customers who buy  on Weekends', fontsize = 30)
plt.xlabel('Weekend or not', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show()

data['VisitorType'].value_counts()

# plotting a pie chart for browsers

size = [10551, 1694, 85]
colors = ['violet', 'magenta', 'pink']
labels = "Returning Visitor", "New_Visitor", "Others"
explode = [0, 0, 0.1]

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Different Typesof Visitors', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()

# visualizing the distribution of customers around the Region

plt.rcParams['figure.figsize'] = (15, 10)
plt.hist(data['TrafficType'], color = 'lightgreen')
plt.title('Distribution of TrafficType in different Regions',fontsize = 30)
plt.xlabel('TrafficType Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# visualizing the distribution of customers around the Region

plt.rcParams['figure.figsize'] = (15, 10)
plt.hist(data['Region'], color = 'lightblue')
plt.title('Distribution of Customers in different Regions',fontsize = 30)
plt.xlabel('Region Codes', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking different no. of browsers they use for shopping

data['Browser'].value_counts()

# plotting a pie chart for browsers

size = [7961, 2462, 736, 467,174, 163, 300]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'blue']
labels = "2", "1","4","5","6","10","others"
explode = [0, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%', startangle = 90)
plt.title('No. of Browsers users browse during shopping', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()

# checking the no. of OSes each user is having

data['OperatingSystems'].value_counts()

# creating a donut chart for the months variations'

# plotting a pie chart for different number of OSes users have.

size = [6601, 2585, 2555, 478, 111]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen']
labels = "2", "1","3","4","others"
explode = [0, 0, 0, 0, 0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Pie Chart for no.of OSes Users have', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()
plt.show()

#checking the months with most no.of customers visiting the online shopping sites

data['Month'].value_counts()

# creating a donut chart for the months variations'

# plotting a pie chart for share of special days

size = [3364, 2998, 1907, 1727, 549, 448, 433, 432, 288, 184]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'magenta', 'lightblue', 'lightgreen', 'violet']
labels = "May", "November", "March", "December", "October", "September", "August", "July", "June", "February"
explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

circle = plt.Circle((0, 0), 0.6, color = 'white')

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Pie Chart for Share of Special Days', fontsize = 30)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()
plt.show()

# looking at the probabilities of special day

data['SpecialDay'].value_counts()

# plotting a pie chart for share of special days

size = [11079, 351, 325, 243, 178, 154]
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan']
labels = "0", "0.6", "0.8", "0.4", "0.2", "1.0"
explode = [0, 0, 0.1, 0, 0, 0.2]

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Pie Chart for Share of Special Days', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.distplot(data['PageValues'], color = 'blue')
plt.title('Variations in Page Values', fontsize = 30)
plt.xlabel('Page Values', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.distplot(data['ExitRates'], color = 'red')
plt.title('Variations in Exit Rates', fontsize = 30)
plt.xlabel('Exit Rates', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.distplot(data['BounceRates'], color = 'red')
plt.title('Variations in Bounce Rates', fontsize = 30)
plt.xlabel('Bounce Rate', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.countplot(data['ProductRelated'].head(20))
plt.title('Product Related Pages Visited by Customers', fontsize = 30)
plt.xlabel('No. of times Product Related Page is visited', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.distplot(data['ProductRelated_Duration'], color = 'violet')
plt.title('Variations in Customers visting Product related pages', fontsize = 30)
plt.xlabel('Product Related Duration', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.countplot(data['Informational'])
plt.title('No. of Informational Pages Visited by Customers', fontsize = 30)
plt.xlabel('Informational Duration', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.distplot(data['Informational_Duration'], color = 'yellow')
plt.title('Variations in Top 40 Administrative duration of different Customers', fontsize = 30)
plt.xlabel('Administrative Duration', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
data['Administrative'].value_counts().plot.bar(color = 'purple')
plt.title('No. of Administrative Pages Visited by Customers', fontsize = 30)
plt.xlabel('Administrative Pages', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# checking the variations in the administrative duration for the online shoppers

plt.rcParams['figure.figsize'] = (15, 10)
sns.distplot(data['Administrative_Duration'], color = 'violet')
plt.title('Variations in Top 40 Administrative duration of different Customers', fontsize = 30)
plt.xlabel('Administrative Duration', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.show()

# product related duration vs revenue

plt.rcParams['figure.figsize'] = (10, 10)
sns.violinplot(data['Revenue'], data['Informational_Duration'], palette = 'colorblind')
plt.title('Info. duration vs Revenue', fontsize = 30)
plt.xlabel('Info. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# product related duration vs revenue

plt.rcParams['figure.figsize'] = (10, 10)
sns.violinplot(data['Revenue'], data['Administrative_Duration'], palette = 'pastel')
plt.title('Admn. duration vs Revenue', fontsize = 30)
plt.xlabel('Admn. duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# product related duration vs revenue

plt.rcParams['figure.figsize'] = (10, 10)
sns.violinplot(data['Revenue'], data['ProductRelated_Duration'], palette = 'dark')
plt.title('Product Related duration vs Revenue', fontsize = 30)
plt.xlabel('Product Related duration', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# exit rate vs revenue

plt.rcParams['figure.figsize'] = (10, 10)
sns.boxplot(data['Revenue'], data['ExitRates'], palette = 'dark')
plt.title('ExitRates vs Revenue', fontsize = 30)
plt.xlabel('ExitRates', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# page values vs revenue

plt.rcParams['figure.figsize'] = (10, 10)
sns.boxplot(data['Revenue'], data['PageValues'], palette = 'pastel')
plt.title('PageValues vs Revenue', fontsize = 30)
plt.xlabel('PageValues', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()


# bounce rates vs revenue

plt.rcParams['figure.figsize'] = (10, 10)
sns.boxplot(data['Revenue'], data['BounceRates'])
plt.title('Bounce Rates vs Revenue', fontsize = 30)
plt.xlabel('Boune Rates', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()

# specialday vs revenue

df = pd.crosstab(data['SpecialDay'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['orange', 'brown'])
plt.title('SpecialDay vs Revenue', fontsize = 30)
plt.show()

# months vs revenue

df = pd.crosstab(data['Month'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['lightgreen', 'black'])
plt.title('Month vs Revenue', fontsize = 30)
plt.show()

# operating system vs Revenue

df = pd.crosstab(data['OperatingSystems'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['pink', 'black'])
plt.title('OperatingSystems vs Revenue', fontsize = 30)
plt.show()

# browser vs Revenue

df = pd.crosstab(data['Browser'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['violet', 'Purple'])
plt.title('Browser vs Revenue', fontsize = 30)
plt.show()

# weekend vs Revenue

df = pd.crosstab(data['Weekend'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['orange', 'crimson'])
plt.title('Weekend vs Revenue', fontsize = 30)
plt.show()

# Traffic Type vs Revenue

df = pd.crosstab(data['TrafficType'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['lightpink', 'yellow'])
plt.title('Traffic Type vs Revenue', fontsize = 30)
plt.show()

# visitor type vs revenue

df = pd.crosstab(data['VisitorType'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['lightgreen', 'green'])
plt.title('Visitor Type vs Revenue', fontsize = 30)
plt.show()


# region vs Revenue

df = pd.crosstab(data['Region'], data['Revenue'])
df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 10), color = ['lightblue', 'blue'])
plt.title('Region vs Revenue', fontsize = 30)
plt.show()

# lm plot

plt.rcParams['figure.figsize'] = (20, 10)

sns.lmplot(x = 'Administrative', y = 'Informational', data = data, x_jitter = 0.05)
plt.title('LM Plot between Admistrative and Information', fontsize = 15)


# month vs pagevalues wrt revenue

plt.rcParams['figure.figsize'] = (15, 10)
sns.boxplot(x = data['Month'], y = data['PageValues'], hue = data['Revenue'], palette = 'dark')
plt.title('Month vs PageValues w.r.t. Revenue', fontsize = 30)
plt.show()

# month vs exitrates wrt revenue

plt.rcParams['figure.figsize'] = (15, 10)
sns.boxplot(x = data['Month'], y = data['ExitRates'], hue = data['Revenue'], palette = 'pastel')
plt.title('Month vs ExitRates w.r.t. Revenue', fontsize = 30)
plt.show()

# month vs bouncerates wrt revenue

plt.rcParams['figure.figsize'] = (15, 10)
sns.boxplot(x = data['Month'], y = data['BounceRates'], hue = data['Revenue'], palette = 'colorblind')
plt.title('Month vs BounceRates w.r.t. Revenue', fontsize = 30)
plt.show()

# visitor type vs exit rates w.r.t revenue

plt.rcParams['figure.figsize'] = (15, 10)
sns.violinplot(x = data['VisitorType'], y = data['BounceRates'], hue = data['Revenue'], palette = 'colorblind')
plt.title('Visitor Type vs BounceRates w.r.t. Revenue', fontsize = 30)
plt.show()

# visitor type vs exit rates w.r.t revenue

plt.rcParams['figure.figsize'] = (15, 10)
sns.violinplot(x = data['VisitorType'], y = data['ExitRates'], hue = data['Revenue'], palette = 'pastel')
plt.title('Visitor Type vs ExitRates w.r.t. Revenue', fontsize = 30)
plt.show()

# visitor type vs exit rates w.r.t revenue

plt.rcParams['figure.figsize'] = (15, 10)
sns.violinplot(x = data['VisitorType'], y = data['PageValues'], hue = data['Revenue'], palette = 'colorblind')
plt.title('Visitor Type vs PageValues w.r.t. Revenue', fontsize = 30)
plt.show()

# region vs pagevalues w.r.t. revenue

plt.rcParams['figure.figsize'] = (15, 10)
sns.barplot(x = data['Region'], y = data['PageValues'], hue = data['Revenue'], palette = 'colorblind')
plt.title('Region vs PageValues w.r.t. Revenue', fontsize = 30)
plt.show()

# region vs exit rates w.r.t. revenue


plt.rcParams['figure.figsize'] = (15, 10)
sns.barplot(x = data['Region'], y = data['ExitRates'], hue = data['Revenue'], palette = 'dark')
plt.title('Region vs Exit Rates w.r.t. Revenue', fontsize = 30)
plt.show()

# region vs Bounce rates wrt revenue


plt.rcParams['figure.figsize'] = (15, 10)
sns.barplot(x = data['Region'], y = data['BounceRates'], hue = data['Revenue'], palette = 'pastel')
plt.title('Region vs Bounce Rates w.r.t. Revenue', fontsize = 30)
plt.show()

data.dtypes
"""
# one hot encoding 

data1 = pd.get_dummies(data)

data1.columns

# label encoding of revenue

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Revenue'] = le.fit_transform(data['Revenue'])
data['Revenue'].value_counts()

# getting dependent and independent variables
x = data1
x = x.drop(['Revenue'], axis = 1)
y = data['Revenue']

# checking the shapes
print("Shape of x:", x.shape)
print("Shape of y:", y.shape)
# splitting the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# checking the shapes
print("Shape of x_train :", x_train.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_test.shape)

# standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# feature extraction using pca
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# feature extraction using ica
from sklearn.decomposition import FastICA
ica =FastICA(n_components=None)
x_train = ica.fit_transform(x_train)
x_test = ica.transform(x_test)

print("logistic regresion..................")
# MODELLING
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# classification report
cr = classification_report(y_test, y_pred)
print(cr)

# MODELLING
print("RandomForestClassifier...........................")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)# classification report
cr = classification_report(y_test, y_pred)
print(cr)
print("AdaBoostClassifier..............................")
# MODELLING
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = AdaBoostClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# classification report
cr = classification_report(y_test, y_pred)
print(cr)
print("DecisionTreeClassifier................")
# MODELLING
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# classification report
cr = classification_report(y_test, y_pred)
print(cr)

print("SVC classifier.....................")
# MODELLING
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# classification report
cr = classification_report(y_test, y_pred)
print(cr)
print("GNB..........................")
# MODELLING
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# classification report
cr = classification_report(y_test, y_pred)
print(cr)

print("KNN.........................")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
model = KNeighborsClassifier(n_neighbors=3,p=2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# classification report
cr = classification_report(y_test, y_pred)
print(cr)
print("extra tree.............................")
from sklearn.ensemble import ExtraTreesClassifier




####################################################################################
# cross validation
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
print("Mean Accuracy :", cvs.mean())
print("Mean Standard Deviation :", cvs.std())

from sklearn.model_selection import GridSearchCV
params = {'C':[0.1, 1, 10], 'kernel':['rbf', 'linear', 'poly'], 'gamma': [0.1, 0.001, 0.0001]}
grid_search = GridSearchCV(estimator = model, param_grid = params, cv = 5)
grid_search = grid_search.fit(x_train, y_train)
print("Best Accuracy :", grid_search.best_score_)
print("Best Parameters :", grid_search.best_params_)

# cross validation
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
print("Mean Accuracy :", cvs.mean())
print("Mean Standard Deviation :", cvs.std())
