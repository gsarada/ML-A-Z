import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

scaler = None
classifier = None

def visualize(X, y):
    X_set, y_set = scaler.inverse_transform(X), y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                        np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
    plt.contourf(X1, X2, classifier.predict(scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(['#FA8072', '#1E90FF']))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(['#FA8072', '#1E90FF'])(i), label = j)
    plt.title("Car Purchase Train")
    plt.xlabel("Age")
    plt.ylabel("Salary")
    #plt.show()

def main(is_linear: False):

    #import teh dataset
    dataset = pd.read_csv("datasets/car_purchase_data.csv")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    #Split the dataset
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)

    #feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    classifier = GaussianNB()
    classifier.fit(X_train_scaled, y_train)

    # Predict new observation
    y_new = classifier.predict(scaler.transform([[29, 80000]]))
    print(f"Predicted value for age 29 and salary 80000 is {y_new}")

    # Predict the test set
    y_pred = classifier.predict(X_test_scaled)
    print(f"Comparison - {np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)),1)}")

    # Analyse accuracy
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy Score - {score}")
    print(f"Confusion matrix - {matrix}")

    #Visualize training set
    #visualize(X_train, y_train)

    #Visualize test set
    #visualize(X_test, y_test)

if __name__ == "__main__":
    print("-------Running linear SVM--------")
    main(True)
    print("-------Running non-linear SVM--------")
    main(False)