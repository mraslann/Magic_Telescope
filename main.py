import pandas as pd
from sklearn import model_selection
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
import random


def read_data():
    return pd.read_csv("magic04 - Copy.csv")


def balance_data(instance):
    num_of_gdata = 12331
    for i in range(5644, 0, -1):
        x = random.randint(0, num_of_gdata)
        instance = instance.drop(instance.index[x])
        num_of_gdata -= 1
    return instance


def classify(x_train, x_test, y_train, y_test):
    classifications = [decision_tree, knn, naive_bayes, random_forest, ada_boost]
    for classification in classifications:
        classification(x_train, x_test, y_train, y_test)


def split_data(instance):
    x = instance.values[:, 0:9]
    y = instance.values[:, 10]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, train_size=0.7,
                                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def calculate_accuracy(y_test, y_pred):
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred, pos_label='g'))
    print("Precision: ", precision_score(y_test, y_pred, pos_label='g'))
    print("F-Score: ", f1_score(y_test, y_pred, pos_label='g'))


def decision_tree(x_train, x_test, y_train, y_test):
    dt_gini = tree.DecisionTreeClassifier(criterion="gini", random_state=42)
    dt_gini.fit(x_train, y_train)
    y_pred_gini = dt_gini.predict(x_test)
    calculate_accuracy(y_test, y_pred_gini)
    dt_entropy = tree.DecisionTreeClassifier(criterion="entropy", random_state=42)
    dt_entropy.fit(x_train, y_train)
    y_pred_entropy = dt_entropy.predict(x_test)
    print("Decision tree:")
    calculate_accuracy(y_test, y_pred_entropy)


def knn(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print("KNN:")
    calculate_accuracy(y_test, y_pred)


def naive_bayes(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    print("Naive Bayes:")
    calculate_accuracy(y_test, y_pred)


def random_forest(x_train, x_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print("Random Forest: ")
    calculate_accuracy(y_test, y_pred)


def ada_boost(x_train, x_test, y_train, y_test):
    ada = AdaBoostClassifier(n_estimators=50)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    print("Ada Boost:")
    calculate_accuracy(y_test, y_pred)


if __name__ == '__main__':
    data = read_data()
    data = balance_data(data)
    x_train, x_test, y_train, y_test = split_data(data)
    classify(x_train, x_test, y_train, y_test)
