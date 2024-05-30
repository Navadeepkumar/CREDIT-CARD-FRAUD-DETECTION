# CREDIT-CARD-FRAUD-DETECTION
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('credit_card_transactions.csv')


df_clean = df.dropna()  # remove rows with missing values
df_clean['Time_STD'] = StandardScaler().fit_transform(df_clean['Time'].values.reshape(-1, 1))
df_clean['Amount_STD'] = StandardScaler().fit_transform(df_clean['Amount'].values.reshape(-1, 1))
df_clean.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df_clean.drop('Class', axis=1)
y = df_clean['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True, random_state=110)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_LR = log_reg.predict(X_test)
y_prob_LR = log_reg.predict_proba(X_test)[:, 1]


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_DT = dt.predict(X_test)
y_prob_DT = dt.predict_proba(X_test)[:, 1]


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_RF = rf.predict(X_test)
y_prob_RF = rf.predict_proba(X_test)[:, 1]


print("Evaluation Metrics Report")
print(classification_report(y_test, y_pred_LR))
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_prob_LR)))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred_LR) * 100))

print("Decision Tree")
print(classification_report(y_test, y_pred_DT))
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_prob_DT)))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred_DT) * 100))

print("Random Forest")
print(classification_report(y_test, y_pred_RF))
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_prob_RF)))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred_RF) * 100))


plt.figure(figsize=(10, 4))
sns.set()
plt.subplot(1, 2, 1)
sns.lineplot(x=[0, 1], y=[0, 1], color='black', linestyle='--')
sns.lineplot(x=y_prob_LR, y=y_test, color='blue', label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')

plt.subplot(1, 2, 2)
sns.lineplot(x=[0, 1], y=[0, 1], color='black', linestyle='--')
sns.lineplot(x=y_prob_DT, y=y_test, color='green', label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')

plt.show()
