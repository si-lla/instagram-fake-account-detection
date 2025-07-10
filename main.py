import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn import tree

# Load the training dataset
df = pd.read_csv('train.csv')
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# 🔽 EDA starts here 🔽

# Plot the distribution of target variable
sns.countplot(x='fake', data=df)
plt.title('Distribution of Fake vs Genuine Accounts')
plt.xlabel('Account Type (0 = Genuine, 1 = Fake)')
plt.ylabel('Count')
plt.show()

# Profile picture presence vs fake/genuine
sns.barplot(x='fake', y='profile pic', data=df)
plt.title("Profile Picture Presence in Fake vs Genuine Accounts")
plt.xlabel("Account Type (0 = Genuine, 1 = Fake)")
plt.ylabel("Avg. Profile Pic Presence (1 = Yes, 0 = No)")
plt.show()

# Followers count vs fake/genuine
sns.boxplot(x='fake', y='#followers', data=df)
plt.title("Followers Count in Fake vs Genuine Accounts")
plt.xlabel("Account Type (0 = Genuine, 1 = Fake)")
plt.ylabel("Number of Followers")
plt.show()

# Following count vs fake/genuine
sns.boxplot(x='fake', y='#follows', data=df)
plt.title("Following Count in Fake vs Genuine Accounts")
plt.xlabel("Account Type (0 = Genuine, 1 = Fake)")
plt.ylabel("Number of Accounts Followed")
plt.show()

# Posts count vs fake/genuine
sns.boxplot(x='fake', y='#posts', data=df)
plt.title("Posts Count in Fake vs Genuine Accounts")
plt.xlabel("Account Type (0 = Genuine, 1 = Fake)")
plt.ylabel("Number of Posts")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Prepare data for model
X = df.drop('fake', axis=1)
y = df['fake']

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Initialize and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot decision tree
plt.figure(figsize=(20, 15))
tree.plot_tree(model, 
               filled=True, 
               feature_names=X.columns, 
               class_names=['Genuine', 'Fake'], 
               rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Feature importance plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Load the test dataset
df_test = pd.read_csv('test.csv')

# Make predictions on the test set
y_test_pred = model.predict(df_test.drop('fake', axis=1))

# Evaluate
print("Test Accuracy:", accuracy_score(df_test['fake'], y_test_pred))
print("\nConfusion Matrix:\n", confusion_matrix(df_test['fake'], y_test_pred))
print("\nClassification Report:\n", classification_report(df_test['fake'], y_test_pred))

# Save predictions to CSV
df_test['predicted_fake'] = y_test_pred
df_test.to_csv('test_with_predictions.csv', index=False)

# Create summary for Tableau
summary = df.groupby('fake').agg({
    'profile pic': 'mean',
    '#followers': 'mean',
    '#follows': 'mean',
    '#posts': 'mean',
    'private': 'mean',
    'external URL': 'mean',
    'description length': 'mean'
}).reset_index()

# Convert binary 'fake' column to labels
summary['Account Type'] = summary['fake'].map({0: 'Genuine', 1: 'Fake'})
summary.drop('fake', axis=1, inplace=True)

# Save to CSV
summary.to_csv('tableau_summary.csv', index=False)
print("Summary exported for Tableau.")

# ✅ Confusion Matrix Plot (Saved)
# Make sure 'images/' folder exists or update the path if needed
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("images/confusion-matrix.png")  # Saves image to /images/
plt.show()
