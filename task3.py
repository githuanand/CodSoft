import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor as xGBRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv(r"C:\Users\ANAND\Downloads\FILE\Customer_Churn_Prediction\Churn_Modelling.csv")

df.head()

(df.isnull().sum() / df.shape[0]*100).sort_values(ascending=False)

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

plt.figure(figsize=(15, 6))

sns.histplot(data=df, x='Age', hue='Exited',  multiple="stack",kde=True, palette="viridis")
plt.title('Age Distribution by Exited Status')

plt.tight_layout()
plt.show()

mode = df['Age'][df['Exited'] == 0].mode()[0]
mean = df['Age'][df['Exited'] == 0].mean()
median = df['Age'][df['Exited'] == 0].median()
mode_exit = df['Age'][df['Exited'] == 1].mode()[0]
mean_exit= df['Age'][df['Exited'] == 1].mean()
median_exit = df['Age'][df['Exited'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics    |  Exited = 0 |  Exited= 1       |")
print("-----------------------------------------------------")
print(f"| Mode             |  {mode:<9}  |  {mode_exit:<14}  |")
print(f"| Median           |  {median:<9}  |  {median_exit:<14}  |")
print(f"| Mean             |  {mean:<9.2f}  |  {mean_exit:<14.2f}  |")
print("-----------------------------------------------------")

q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)

q1_exit_0 = df['Age'][df['Exited'] == 0].quantile(0.25)
q3_exit_0 = df['Age'][df['Exited'] == 0].quantile(0.75)

q1_exit_1 = df['Age'][df['Exited'] == 1].quantile(0.25)
q3_exit_1 = df['Age'][df['Exited'] == 1].quantile(0.75)


print("Quartiles for Age Distribution when Exited = 0:")
print(f"Q1: {q1_exit_0}, Q3: {q3_exit_0}\n")

print("Quartiles for Age Distribution when Exited = 1:")
print(f"Q1: {q1_exit_1}, Q3: {q3_exit_1}")

df_exited = df[df['Exited'] == 1]

plt.figure(figsize=(15, 6))

sns.histplot(data=df_exited, x='Age', hue='HasCrCard', multiple="stack", kde=True, palette="viridis")

plt.title('Age Distribution by Credit Card Ownership for Exited Individuals')
plt.xlabel('Age')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

mode = df_exited['Age'][df['HasCrCard'] == 0].mode()[0]
mean = df_exited['Age'][df['HasCrCard'] == 0].mean()
median = df_exited['Age'][df['HasCrCard'] == 0].median()
mode_exit = df_exited['Age'][df['HasCrCard'] == 1].mode()[0]
mean_exit= df_exited['Age'][df['HasCrCard'] == 1].mean()
median_exit = df_exited['Age'][df['HasCrCard'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics  |  HasCrCard = 0 |  HasCrCard= 1   |")
print("-----------------------------------------------------")
print(f"| Mode           |  {mode:<9}  |  {mode_exit:<14}    |")
print(f"| Median         |  {median:<9}  |  {median_exit:<14}    |")
print(f"| Mean           |  {mean:<9.2f}  |  {mean_exit:<14.2f}    |")
print("-----------------------------------------------------")

plt.figure(figsize=(15, 6))

sns.histplot(data=df, x='CreditScore', hue='Exited',  multiple="stack",kde=True,  palette="viridis")
plt.title('Credit Score Distribution by Exited Status')

plt.tight_layout()
plt.show()

mode = df['CreditScore'][df['Exited'] == 0].mode()[0]
mean = df['CreditScore'][df['Exited'] == 0].mean()
median = df['CreditScore'][df['Exited'] == 0].median()
mode_exit = df['CreditScore'][df['Exited'] == 1].mode()[0]
mean_exit= df['CreditScore'][df['Exited'] == 1].mean()
median_exit = df['CreditScore'][df['Exited'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics    |  Exited = 0 |  Exited= 1       |")
print("-----------------------------------------------------")
print(f"| Mode             |  {mode:<9}  |  {mode_exit:<14}  |")
print(f"| Median           |  {median:<9}  |  {median_exit:<14}  |")
print(f"| Mean             |  {mean:<9.2f}  |  {mean_exit:<14.2f}  |")
print("-----------------------------------------------------")

q1 = df['CreditScore'].quantile(0.25)
q3 = df['CreditScore'].quantile(0.75)

q1_exit_0 = df['CreditScore'][df['Exited'] == 0].quantile(0.25)
q3_exit_0 = df['CreditScore'][df['Exited'] == 0].quantile(0.75)

q1_exit_1 = df['CreditScore'][df['Exited'] == 1].quantile(0.25)
q3_exit_1 = df['CreditScore'][df['Exited'] == 1].quantile(0.75)


print("Quartiles for Age Distribution when Exited = 0:")
print(f"Q1: {q1_exit_0}, Q3: {q3_exit_0}\n")

print("Quartiles for Age Distribution when Exited = 1:")
print(f"Q1: {q1_exit_1}, Q3: {q3_exit_1}")


plt.figure(figsize=(15, 6))

sns.histplot(data=df, x='Gender', hue='Exited',  multiple="stack", palette="viridis")
plt.title('Gender Distribution by Exited Status')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
sns.histplot(data=df.loc[df['Exited'] == 0], x="EstimatedSalary", kde=True, color=sns.color_palette("viridis")[1])
plt.title("Estimated Salary Distribution for Not Exited")

plt.subplot(1, 2, 2)
sns.histplot(data=df.loc[df['Exited'] == 1], x="EstimatedSalary", kde=True, color=sns.color_palette("viridis")[1])
plt.title("Estimated Salary Distribution for Exited")

plt.tight_layout()
plt.show()

mode = df['EstimatedSalary'][df['Exited'] == 0].mode()[0]
mean = df['EstimatedSalary'][df['Exited'] == 0].mean()
median = df['EstimatedSalary'][df['Exited'] == 0].median()
mode_exit = df['EstimatedSalary'][df['Exited'] == 1].mode()[0]
mean_exit= df['EstimatedSalary'][df['Exited'] == 1].mean()
median_exit = df['EstimatedSalary'][df['Exited'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics    |  Exited = 0 |  Exited= 1       |")
print("-----------------------------------------------------")
print(f"| Mode             |  {mode:<9}  |  {mode_exit:<14}  |")
print(f"| Median           |  {median:<9}  |  {median_exit:<14.2f}  |")
print(f"| Mean             |  {mean:<9.2f}  |  {mean_exit:<14.2f}  |")
print("-----------------------------------------------------")

plt.figure(figsize=(15, 6))

sns.histplot(data=df, x='IsActiveMember', hue='Exited',  multiple="stack",kde=True,  palette="viridis")
plt.title('Is Active Member Distribution by Exited Status')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))

sns.histplot(data=df, x='NumOfProducts', hue='Exited',  multiple="stack",kde=True,  palette="viridis")
plt.title('Num Of Products Distribution by Exited Status')

plt.tight_layout()
plt.show()

churn_count = df[df['Exited'] == 1].groupby('Geography').size().reset_index(name='churn_count')

non_churn_count = df[df['Exited'] == 0].groupby('Geography').size().reset_index(name='non_churn_count')

combined_count = churn_count.merge(non_churn_count, on='Geography')

total_count = df['Geography'].value_counts().reset_index()
total_count.columns = ['Geography', 'total_count']
combined_count = combined_count.merge(total_count, on='Geography')
combined_count['churn_percentage'] = (combined_count['churn_count'] / combined_count['total_count']).round(4) * 100

combined_count
df['Geography'] = df['Geography'].map({'France': 1, 'Germany': 2, 'Spain': 3})

df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

df.head()
X_train, X_test, y_train, y_test =train_test_split(df.drop('Exited',axis=1),df['Exited'],test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

modelo_tree = DecisionTreeClassifier(max_depth=5)

modelo_tree.fit(X_train,y_train)

acc_tree = round(modelo_tree.score(X_train,y_train) * 100, 2)
predict_tree=modelo_tree.predict(X_test)
print("Accuracy of the Decision Tree Classifie model is: {}".format(acc_tree))


df['Geography'] = df['Geography'].map({'France': 1, 'Germany': 2, 'Spain': 3})
