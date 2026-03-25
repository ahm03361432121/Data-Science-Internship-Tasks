(Iris Dataset Visualization)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('iris')
print(df.shape)
print(df.columns)
df.head()
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=df)
plt.show()
df['petal_length'].hist()
plt.show()
sns.boxplot(x='species', y='sepal_length', data=df)
plt.show()

(Credit Risk Prediction)
from google.colab import files
uploaded = files.upload()
import pandas as pd
df = pd.read_csv('loan_data.csv')  
df.head()
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
    from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df.columns = df.columns.str.strip()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

(Insurance Prediction)
from google.colab import files
import pandas as pd
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)
df.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])
X = df.drop('charges', axis=1)
y = df['charges']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE:", mean_absolute_error(y, y_pred))
import numpy as np
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
