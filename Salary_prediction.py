import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import  train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# import dataset 
dataset = pd.read_csv('IT Salary EU 2018.csv')
dataset = dataset.dropna(axis=0, subset=['Position','Years of experience', 'Your level', 'Current Salary'])
dataset = dataset.loc[:, ~dataset.columns.isin(['Main language at work', 'City', 'Timestamp', 'Gender', 'Are you getting any Stock Options?', 'Company type', 'Company size'])]

def fill_na_df(df):
    year_ago = dataset.groupby(['Your level'],as_index=False)['Salary one year ago'].mean()
    two_year_ago = dataset.groupby(['Your level'],as_index=False)['Salary two years ago'].mean()
    age = dataset.groupby(['Your level'], as_index=False)['Age'].mean()
    for index, row in dataset.iterrows():
        level = row['Your level']
        if np.isnan(row['Salary one year ago']):
            df.at[index ,'Salary one year ago'] = year_ago.loc[year_ago['Your level'] == level]['Salary one year ago']
        if np.isnan(row['Salary two years ago']):
            df.at[index, 'Salary two years ago'] = two_year_ago.loc[two_year_ago['Your level'] == level]['Salary two years ago']
        if np.isnan(row['Age']):
            df.at[index, 'Age'] = age.loc[age['Your level'] == level]['Age']

fill_na_df(dataset)

# visualization data
plt.hist(x= dataset['Age'], edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('./visualization/Age_Distribution.jpeg')
plt.show()

year_df = dataset.groupby(['Years of experience'])['Years of experience'].agg([len]).reset_index()
plt.bar(year_df['Years of experience'], year_df['len'], edgecolor='black')
plt.title('Year of Experiences Distribution')
plt.xlabel('Experiences (Year)')
plt.ylabel('Frequency')
plt.savefig('./visualization/Year_Distribution.jpeg')
plt.show()

salary_binwith = 10000
plt.hist(dataset['Current Salary'], bins=range( int(np.floor(min(dataset['Current Salary']))), int(np.ceil(max(dataset['Current Salary'])))+ salary_binwith, salary_binwith ), edgecolor='black')
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.savefig('./visualization/Salary_Distribution.jpeg')
plt.show()

# data preprocessing

X = dataset.loc[:, dataset.columns != 'Current Salary'].values
y = dataset.loc[:, 'Current Salary'].values

kf = KFold(n_splits=6, random_state=1, shuffle=True)
kf.get_n_splits(X)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean' )
oe = OrdinalEncoder(handle_unknown='ignore')

linear_model = LinearRegression()

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train[:, [1,3]] = oe.fit_transform(X_train[:, [1,3]])
    X_test[:, [1,3]] = oe.transform(X_test[:, [1,3]])
    linear_model.fit(X_train,y_train)

    salary_predicts = linear_model.predict(X_test)
    salary_predicts = salary_predicts.reshape(len(salary_predicts), 1)
    y_test = salary_predicts.reshape(len(y_test), 1)
    # np.printoptions(precision=2)
    # result = np.concatenate((salary_predicts, y_test), axis= 1)

print(linear_model.predict([[40, 281.0, 9.0, 1, 50000.0, 45000]]))
