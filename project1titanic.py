# ======================================
# Titanic Survival Prediction Project
# ======================================

# Import libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# ==============================
# 1. Load Dataset
# ==============================

# Load Titanic dataset from seaborn
df = sns.load_dataset("titanic")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Check missing values
print(df.isnull().sum())

# ==============================
# 2. Exploratory Data Analysis
# ==============================

# Survival distribution
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

# Survival by gender
sns.barplot(x='sex', y='survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

# Survival by passenger class
sns.barplot(x='pclass', y='survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

# ==============================
# 3. Data Preprocessing
# ==============================

# Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(columns=[
    'deck',
    'class',
    'who',
    'adult_male',
    'alive',
    'embark_town',
    'alone'
], inplace=True)

# Convert categorical variables using One-Hot Encoding
df = pd.get_dummies(df, columns=['sex','embarked'], drop_first=True)

# ==============================
# 4. Feature / Target Split
# ==============================

X = df.drop('survived', axis=1)
y = df['survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling (for age and fare)
scaler = StandardScaler()
num_cols = ['age','fare']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ==============================
# 5. Model Training & Evaluation
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print("="*50)
    print(name)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==============================
# 6. Model Comparison
# ==============================

print("\nModel Accuracy Comparison:")
for k, v in results.items():
    print(f"{k} : {v:.4f}")

plt.figure(figsize=(8,4))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()
