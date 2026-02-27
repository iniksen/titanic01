# ============================================================
#  Titanic Survival Prediction Project
#  Classification - Full Workflow
# ============================================================

# ============================================================
#  مرحله 0: کتابخانه‌های مورد نیاز
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

# ============================================================
#  مرحله 1: بارگذاری و شناخت اولیه داده‌ها
# ============================================================

print("\n================= مرحله 1: شناخت داده =================")

# 1.1 بارگذاری دیتاست
df = sns.load_dataset("titanic")

# 1.2 ابعاد دیتاست
print("Shape of dataset:", df.shape)

# 1.3 مشاهده 5 سطر اول
print("\nHead of dataset:")
print(df.head())

# 1.4 اطلاعات ستون‌ها
print("\nDataset Info:")
print(df.info())

# 1.5 آمار توصیفی
print("\nDescriptive Statistics:")
print(df.describe())

# ============================================================
#  مرحله 2: تحلیل اکتشافی داده‌ها (EDA)
# ============================================================

print("\n================= مرحله 2: EDA =================")

# -----------------------------
# 2.1 مدیریت داده‌های گمشده
# -----------------------------

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# پر کردن age با median
df['age'].fillna(df['age'].median(), inplace=True)

# پر کردن embarked با mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# حذف deck
df.drop(columns=['deck'], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# 2.2 تحلیل تک‌متغیره
# -----------------------------

# متغیر هدف
plt.figure()
sns.countplot(x='survived', data=df)
plt.title("Distribution of Survival")
plt.show()

# متغیرهای دسته‌ای
categorical_cols = ['pclass','sex','sibsp','parch','embarked']

for col in categorical_cols:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.show()

# متغیرهای عددی
plt.figure()
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure()
sns.histplot(df['fare'], kde=True)
plt.title("Fare Distribution")
plt.show()

# -----------------------------
# 2.3 تحلیل دومتیغیره
# -----------------------------

plt.figure()
sns.barplot(x='sex', y='survived', data=df)
plt.title("Survival Rate by Sex")
plt.show()

plt.figure()
sns.barplot(x='pclass', y='survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

plt.figure()
sns.barplot(x='embarked', y='survived', data=df)
plt.title("Survival Rate by Embarked Port")
plt.show()

plt.figure()
sns.violinplot(x='survived', y='age', data=df)
plt.title("Age Distribution by Survival")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ============================================================
#  مرحله 3: پیش‌پردازش و مهندسی ویژگی
# ============================================================

print("\n================= مرحله 3: Preprocessing =================")
# ============================================================
#  مرحله 3: Preprocessing 
# ============================================================

# 3.1 حذف ستون‌های غیرضروری
drop_cols = ['deck','class','who','adult_male',
             'alive','embark_town']

df.drop(columns=[col for col in drop_cols if col in df.columns],
        inplace=True)

# 3.2 تبدیل تمام ستون‌های متنی به عددی (روش حرفه‌ای و امن)

df = pd.get_dummies(df, drop_first=True)

# بررسی اینکه ستون متنی باقی نمانده باشد
print("\nColumn Types After Encoding:")
print(df.dtypes)

# 3.3 جداسازی X و y
X = df.drop('survived', axis=1)
y = df['survived']

# تقسیم داده
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3.4 Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPreprocessing Completed Successfully ✅")

# ============================================================
#  مرحله 4: آموزش و ارزیابی مدل‌ها
# ============================================================

print("\n================= مرحله 4: Model Training =================")

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

results = {}

for name, model in models.items():

    print(f"\n================ {name} ================")

    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = np.mean(y_pred == y_test)
    results[name] = acc

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# ============================================================
#  مرحله 5: مقایسه مدل‌ها
# ============================================================

print("\n================= مرحله 5: Model Comparison =================")

results_df = pd.DataFrame(results.items(), columns=["Model","Accuracy"])
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print(results_df)

plt.figure()
sns.barplot(x="Accuracy", y="Model", data=results_df)
plt.title("Model Accuracy Comparison")
plt.show()

print("\nپروژه با موفقیت اجرا شد ✅")