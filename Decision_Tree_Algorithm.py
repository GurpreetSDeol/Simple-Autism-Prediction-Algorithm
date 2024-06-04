from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data 

file_path = 'Encoded_and_Cleaned_Autism_Adult_Data.csv'
df = pd.read_csv(file_path)

# Split the data 
features = df.drop('Class/ASD', axis=1) 
target = df['Class/ASD']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=6)


# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=6)

# Train the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))