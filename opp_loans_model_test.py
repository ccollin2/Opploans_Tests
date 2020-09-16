#Christopher Collins Python Model Test, Opploans

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#I was having trouble using the xgboost package.  
#It is not a library that I am familiar with and 
#I was having trouble installing the software on 
#my current system.  I am probably due for a new
#laptop given that mine is from 2013.  Regardless,
#I did my best to complete the exercise to showcase
#my programming skills in data modelling.  Thank you.  

pd.set_option('display.max_columns', None) #ensure entire df is displayed
df = pd.read_csv('Lending_Club_v2.csv', dtype='object') #read in data

def remove_nulls(df, axis = 1, percent=0.3): #define function to clean data
    df = df.copy()
    ishape = df.shape
    if axis == 0:
        rownames = df.transpose().isnull().sum()
        rownames = list(rownames[rownames.values > percent*len(df)].index)
        df.drop(df.index[rownames],inplace=True) 
        print("\nNumber of Rows dropped\t: ",len(rownames))
    else:
        colnames = (df.isnull().sum()/len(df))
        colnames = list(colnames[colnames.values>=percent].index)
        df.drop(labels = colnames,axis =1,inplace=True)        
        print("Number of Columns dropped\t: ",len(colnames))
        
    print("\nOld dataset rows,columns",ishape,"\nNew dataset rows,columns",df.shape)

    
    return df
    
df = remove_nulls(df, axis = 1, percent=0.3) #remove columns where null values are >= 30%
df = remove_nulls(df, axis = 0, percent=0.3) #remove rows where null values >= 30%   

unique = df.nunique()
unique = unique[unique.values == 1] #prepare to drop values data where there is only 1 unique value

df.drop(labels = list(unique.index), axis =1, inplace=True)
print('Now left with', df.shape, 'rows & columns.')


def handle_non_numerical_data(df): #convert non-numerical data
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

#perform principal component analysis
array = df.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

#prepare data for modeling
loan = df.to_numpy()
loan = df.apply(pd.to_numeric, errors='coerce')
loan = loan.dropna()
array = loan.values
#print(loan)
X = array[:,0:8]
y = array[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


#begin with linear regression to see if the data is linear
clf = LinearRegression(n_jobs=-1) #threading with n_jobs=-1 will use all available threads
clf.fit(X_train, y_train) #fitting our training features, trained
confidence = clf.score(X_test, y_test) #test
print('LinearRegression Confidence Interval:', confidence)

#continue with KNN in attempt to classify whole dataset
clf = neighbors.KNeighborsClassifier() #define classifier
clf.fit(X_train, y_train) #train the classifier
accuracy = clf.score(X_test, y_test) #test
print('KNN Confidence Interval:', accuracy)

#After running these algorithms with the full feature set
#We are now going to try running a decision tree with a
#specific set of features for a more useful analysis
#print(df.head())

#I chose this set of features becuase I felt they had a 
#relationship with our target variable of loan status
feature_cols = ['funded_amnt','term','int_rate','installment',
'sub_grade','emp_length','annual_inc','dti','pub_rec',
'total_pymnt','delinq_amnt','pub_rec_bankruptcies'] #set of features

X = df[feature_cols]
y = df.loan_status #target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
clf = DecisionTreeClassifier() #create classifier
clf = clf.fit(X_train,y_train) #train classifier
y_pred = clf.predict(X_test) #predict response for test data

#model accuracy
print("Accuracy Decision Tree:",metrics.accuracy_score(y_test, y_pred))

#attempt to optimize decision tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy Optimized Decision Tree:",metrics.accuracy_score(y_test, y_pred))



