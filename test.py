from sklearn.model_selection import train_test_split
import pandas as pd

# implement a Naive Bayes classifier from scratch.
# 1. Create a classifier object.
# 2. Train the classifier on the training data.
# 3. Test the classifier on the test data.
# 4. Report the accuracy of the classifier.


df = pd.read_csv("iris.data")

y = df.iloc[:, 4].values
X = df.iloc[:, 0:4].values
# split the dataset randomly in two subsets (70% / 30%).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, shuffle=True)
