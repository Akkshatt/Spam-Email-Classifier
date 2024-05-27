import pandas as pd
import numpy as np

class NBClassifier:
    def __init__(self, b=1):
        self.cp = {}
        self.wgsp = {}
        self.wgnsp = {}
        self.bias = b

    def fit(self, X, y, wl):
        num_c = sum(y)
        num_nc = len(y) - num_c
        ts = len(y)
        self.cp['c'] = np.log(num_c / ts) if num_c > 0 else 0
        self.cp['nc'] = np.log(num_nc / ts) if num_nc > 0 else 0
        
        c_samples = X[y == 1]
        nc_samples = X[y == 0]
        
        c_samples = c_samples[wl]
        nc_samples = nc_samples[wl]
        
        self.wgsp = np.log((c_samples.sum(axis=0) + self.bias) / (c_samples.sum().sum() + len(c_samples.columns) * self.bias))
        self.wgnsp = np.log((nc_samples.sum(axis=0) + self.bias) / (nc_samples.sum().sum() + len(nc_samples.columns) * self.bias))

    def predict(self, X):
        preds = []
        for _, s in X.iterrows():
            c_score = self.cp['c']
            nc_score = self.cp['nc']
            
            c_score += np.dot(s, self.wgsp)
            nc_score += np.dot(s, self.wgnsp)

            if c_score > nc_score:
                preds.append(1)
            else:
                preds.append(0)

        return preds

# Assuming the remaining code for loading the dataset, splitting it into train and test sets, and evaluating the model remains the same.

# Load data
data = pd.read_csv('emails.csv')

# Assuming you correctly set the index of the Prediction column
pci = data.columns.get_loc('Prediction')
wl = data.columns[1:pci]

# Split data into features and target
X = data.iloc[:, 1:pci]
y = data.iloc[:, pci]

# Split data into train and test sets
X_train = X.iloc[:4500]
y_train = y.iloc[:4500]
X_test = X.iloc[4500:]
y_test = y.iloc[4500:]

# Initialize and train the classifier
clf = NBClassifier()
clf.fit(X_train, y_train, wl)

# Make predictions
preds = clf.predict(X_test)

# Evaluate the model
cm = pd.crosstab(y_test, preds, rownames=['Actual'], colnames=['Predicted'])
cc = cm[1][1]
cnc = cm[0][0]
a = (cc + cnc) / len(y_test)

# Print evaluation metrics
print("Training data")
print(X_train,'\n')

print("Testing data")
print(X_test)

print("\nCorrect category samples predicted =", cc)
print("Correct not category samples predicted =", cnc)
print("Undecidable samples =", cm.sum().sum() - cc - cnc)
print("Actual category samples =", y_test.sum())
print("Actual not category samples =", len(y_test) - y_test.sum())
print("Accuracy =", a * 100, "%")
