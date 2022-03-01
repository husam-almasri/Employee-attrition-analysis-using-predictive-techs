#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement

# A large company named XYZ, employs, at any given point of time, around 4500 employees. However, every year, around 15% of its employees leave the company and need to be replaced with the talent pool available in the job market. The management believes that this level of attrition (employees leaving, either on their own or because they got fired) is bad for the company, because of the following reasons.
# 
# - The former employees’ projects get delayed, which makes it difficult to meet timelines, resulting in a reputation loss among consumers and partners.
# 
# - A sizeable department has to be maintained, for the purposes of recruiting new talent.
# - More often than not, the new employees have to be trained for the job and/or given time to acclimatise themselves to the company Hence, the management has contracted an HR analytics firm to understand what factors they should focus on, in order to curb attrition. In other words, they want to know what changes they should make to their workplace, in order to get most of their employees to stay. Also, they want to know which of these variables is most important and needs to be addressed right away.

# #### Goal of the case study

# Model  the probability of attrition using a logistic regression. The results thus obtained will be used by the management to understand what changes they should make to their workplace, in order to get most of their employees to stay.

# ### Project Lificycle

# - Import & Merge Datasets
# - Missing Value's Imputation
# - Exploratory Data Analysis
# - Feature Engineering
# - Model Building
# - Model Evaluation

# In[ ]:


# !pip install plotly


# Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
# import plotly as py
get_ipython().run_line_magic('matplotlib', 'inline')
# Change pandas settings to Display all columns
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# #### Import & Merge Datasets

# The project has 5 datasets. This dataset is for the entire year 2015.

# In[ ]:


# Employees check in and out data
check_in_data=pd.read_csv('data sets/in_time.csv',index_col=[0])
check_out_data=pd.read_csv('data sets/out_time.csv',index_col=[0])
# Survey data:
# Survey given by the manager of every employee
manager_survey_data=pd.read_csv('data sets/manager_survey_data.csv',index_col=['EmployeeID'])
# Survey given by employee.
employee_survey_data=pd.read_csv('data sets/employee_survey_data.csv',index_col=['EmployeeID'])
# Employee work related general details.
general_data=pd.read_csv('data sets/general_data.csv',index_col=['EmployeeID'])


# In[ ]:


check_in_data.head()


# In[ ]:


# Drop Vacation Days, where all employees are off
check_in_data=check_in_data.dropna(axis=1,how='all')
check_out_data=check_out_data.dropna(axis=1,how='all')


# Missing values in check_in_data and check_out_data means that the employee didn't show up that day.

# In[ ]:


# Fill the missing values with 0
check_in_data.fillna(0,inplace=True)
check_in_data.head(1)


# In[ ]:


check_out_data.fillna(0,inplace=True)
check_out_data.head(1)


# In[ ]:


check_in_data.info()


# In[ ]:


check_out_data.info()


# Convert all columns to Datetime format

# In[ ]:


check_in_data=check_in_data.apply(pd.to_datetime,errors="raise")
check_out_data=check_out_data.apply(pd.to_datetime,errors="raise")


# In[ ]:


check_out_data.info()


# Find actual work hours per employee per day

# In[ ]:


in_out_diff=pd.DataFrame()


# In[ ]:


cols=check_in_data.columns
for col in cols:
    in_out_diff[col]=((pd.to_datetime(check_out_data[col].astype(str)) - 
                             pd.to_datetime(check_in_data[col].astype(str))).dt.total_seconds() / 3600.0)


# Find the number of days where the employee didn't show up during 2015.

# In[ ]:


in_out_diff['absences_days']=(in_out_diff == 0).astype(int).sum(axis=1)


# Find mean actual work hours per employee for 2015.

# In[ ]:


in_out_diff['mean_actual_hours']=round(in_out_diff.astype(int).mean(axis=1),2)


# Drop all days and keep 'absences_days' and 'mean_actual_hours' features.

# In[ ]:


in_out_diff=in_out_diff[['absences_days','mean_actual_hours']]
in_out_diff.head()


# Explore general_data dataset.

# In[ ]:


general_data.shape


# Check the duplication

# In[ ]:


general_data[general_data.duplicated()].shape


# From the dupliaction check for the general dataset, it gives us that there is around two-third of the data is duplicated.

# To make sure if the data is really duplicated or just coincidence in employees data similarity, will check the duplication in the Check_in dataset.

# In[ ]:


check_in_data[check_in_data.duplicated()].shape


# After ensured that the duplicated data is just coincidence in employees data similarity, I decided to keep the general_data without droping the duplicated data.
# If you decided to drop the duplicated data, you can run the bellow cell.

# In[ ]:


# general_data.drop_duplicates(subset=None, keep='first', inplace=True)
# general_data[general_data.duplicated()]


# Merge all 4 dataframers.

# In[ ]:


df0=pd.concat([general_data,in_out_diff,manager_survey_data,employee_survey_data], axis=1, ignore_index=False)


# In[ ]:


df0.head()


# Delete null values in the new dataframe, useful if you decided to drop the duplicated data in general_data.

# In[ ]:


df1=df0.copy()
df1.dropna(subset=['Age'],inplace=True)
len(df1)


# Check the distribution of numerical features

# In[ ]:


df1.describe().T


# STD of 'EmployeeCount' and 'StandardHours' is 0, means that all entries have the same value.

# Find the categorical features that have no change in its entities 

# In[ ]:


df1.info()


# In[ ]:


categorical_cols = df1.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if len(df1[col].value_counts())==1:
        print(col+" has "+str(len(df1[col].value_counts()))+ " unique value.")


# Drop 'EmployeeCount', 'StandardHours', 'Over18' features, because they do not contribute any information for the model.

# In[ ]:


df2=df1.copy()
df2.drop(columns = {'EmployeeCount','Over18','StandardHours'}, axis = 1 ,inplace = True)


# In[ ]:


df2.head()


# ####  Missing Value's Imputation

# Check for missing Values

# In[ ]:


df3=df2.copy()
df3[df3.columns[df3.isnull().any()]].isnull().sum()


# In[ ]:


df3.fillna(round(df3.median()),inplace=True)


# #### Exploratory Data Analysis

# In[ ]:


df4=df3.copy()


# In[ ]:


# Correlation matrix
corr = df4.corr()
plt.figure(figsize = (20,20))
sns.heatmap(corr,cbar=True)
plt.show()


# In[ ]:


# Plot the scatter and density for all features
df_temp= df4.select_dtypes(include =[np.number]) # keep only numerical columns
ax = pd.plotting.scatter_matrix(df_temp, alpha=0.75, figsize=[20, 20], diagonal='kde')
corrs = df_temp.corr().values
for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
    ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=10)
plt.suptitle('Scatter and Density Plot')
plt.show()


# Create df_eda, which is a copy of the main dataframe for performing EDA as changed a some features values from number ranks to categorical ranks for a clear plotting

# In[ ]:


df_eda=df4.copy()
# 
df_eda['Education'].replace({
1: 'Below College',
2: 'College',
3: 'Bachelor',
4: 'Master',
5: 'Doctor'
},inplace=True)
# 
df_eda['PerformanceRating'].replace({
1: 'Low',
2: 'Good',
3: 'Excellent',
4: 'Outstanding'
},inplace=True)
# 
df_eda['WorkLifeBalance'].replace({
1: 'Bad',
2: 'Good',
3: 'Better',
4: 'Best'
},inplace=True)
# 
for i in ['EnvironmentSatisfaction', 'JobInvolvement' , 'JobSatisfaction']:
    df_eda[i].replace({
    1: 'Low',
    2: 'Medium',
    3: 'High',
    4: 'Very High'
    },inplace=True)
# 
df_eda.Age = pd.cut(df_eda.Age, range(10, 70, 10)) # Create buckets of 10 years
# df_eda.TotalWorkingYears = pd.cut(df_eda.TotalWorkingYears, range(0, 40, 5)) # Create buckets of 5 years
# df_eda.YearsAtCompany = pd.cut(df_eda.YearsAtCompany, range(0, 40, 5)) # Create buckets of 5 years
# df_eda.YearsSinceLastPromotion = pd.cut(df_eda.YearsSinceLastPromotion, range(0, 15, 3)) # Create buckets of 3 years
# df_eda.YearsWithCurrManager = pd.cut(df_eda.YearsWithCurrManager, range(0, 20, 4)) # Create buckets of 4 years

df_eda.head(50)


# In[ ]:


# Plot function takes multi-features as a list (path) and makes an interactive multi-layers pie-chart
def sunburst_chart(df,path):
# df=> dataset
# path=> parameter corresponding to a list of columns.
#     create datasets corresponding to (Attrition) feature.
    df_temp = df.groupby(path)['Attrition'].count().reset_index()
    df_temp_yes = df[df['Attrition']=="Yes"].groupby(path)['Attrition'].count().reset_index()
    df_temp_no = df[df['Attrition']=="No"].groupby(path)['Attrition'].count().reset_index()

    Attrition_no = pd.merge(df_temp_no, df_temp, on=path, how='inner')
    Attrition_no['Attrition']='Attrition_no'
    Attrition_no['%Attrition']=round((Attrition_no['Attrition_x']/Attrition_no['Attrition_y'])*100)
    Attrition_no.drop(columns={'Attrition_x','Attrition_y'},inplace=True)

    Attrition_yes = pd.merge(df_temp_yes, df_temp, on=path, how='inner')
    Attrition_yes['Attrition']='Attrition_yes'
    Attrition_yes['%Attrition']=round((Attrition_yes['Attrition_x']/Attrition_yes['Attrition_y'])*100)
    Attrition_yes.drop(columns={'Attrition_x','Attrition_y'},inplace=True)

    result = pd.concat([Attrition_yes, Attrition_no]).reset_index(drop=True)
    # sunburst chart
    fig = px.sunburst(result, path=result.columns, values='%Attrition',
                      title= ' around the world',
                      height=620,template='none')
    fig.show()


# In[ ]:


sunburst_chart(df_eda,['Department','JobRole'])


# - In Human resources department, Laboratory technicians and Sales executives have the largest percentage of attrition with with 46% and 44% respectively. At the same time Research scientists have the less percentage of attrition with just 8%.
# 
# - The attrition in Research and development department is low. Research directors have the largest percentage of attrition with 27%.
# 
# - The attrition in Sales department is low. Healthcare represintave have the largest percentage of attrition with 27%.

# In[ ]:


sunburst_chart(df_eda,['Education', 'EducationField'])


# - Except the Diploma holders, the Bachelor and postgraduate in Human resources specialist holders have the largest percentage of attrition, especially PhD holder which all of them leave the company.

# In[ ]:


sunburst_chart(df_eda,['Gender', 'MaritalStatus','Age'])


# - The female divorced and married employees have less attrition than the single employees, maybe because they have more life responsibilities than the singles.
# - The female single employees who are less than 21 years old have the most ability to leave the company, maybe because most of them are interns or doesn't find the right work-invironment for them yet.
# - the male employees have almost the same observations.
# 

# In[ ]:


sunburst_chart(df_eda,['JobRole','JobLevel'])


# - the Manufacturing directors and Sales representatives who have job-level 4, has 40% and 50% of attrition respectively.
# 
# - The highest attrition percentages in all job-rolls appeare in job-level 4.

# In[ ]:


sunburst_chart(df_eda,['StockOptionLevel'])


# - No clear trend in attrition with respect to stock option levels offered to the employees.

# In[ ]:


sunburst_chart(df_eda,['NumCompaniesWorked'])


# - No clear trend in attrition with respect to number of companies an employee has worked in.

# In[ ]:


def bar_chart(df,feature):
    fig = px.histogram(df, x=feature,
                 color='Attrition', barmode='group',text_auto=True,
                 height=400)
    fig.show()


# In[ ]:


bar_chart(df_eda,'BusinessTravel')


# The employees who travel frequently have the largest percentage of attrition, maybe because of the exhaustion caused by business travel.

# In[ ]:


bar_chart(df_eda,'WorkLifeBalance')


# - High attrition percentage among employees have bad work-life balance.

# In[ ]:


bar_chart(df_eda,'JobSatisfaction')


# - More satisfaction of work, less attrition percentage

# In[ ]:


bar_chart(df_eda,'EnvironmentSatisfaction')


# - More satisfaction of work environment, less attrition percentage

# In[ ]:


bar_chart(df_eda,'PerformanceRating')


# - No clear trend in attrition with respect to Performance Rating.

# In[ ]:


bar_chart(df_eda,'JobInvolvement')


# - Employees have low involvement in the work environment have a higher attrition percentage.

# In[ ]:


# Plot function to plot area-charts for the continuous features
def area_chart(df,feature):
    ax=sns.kdeplot(df[df_eda['Attrition']=='Yes'][feature],
             shade=True,label='Attrition = Yes')
    ax=sns.kdeplot(df[df_eda['Attrition']=='No'][feature],
                 shade=True,label='Attrition = No')
    ax.legend()


# In[ ]:


area_chart(df_eda,'MonthlyIncome')


# - It is clear that, employees who have higher income, stay at company longer.

# In[ ]:


area_chart(df_eda,'PercentSalaryHike')


# - It is clear that, employees who have higher increasing in their income, stay at company longer.

# In[ ]:


area_chart(df_eda,'TrainingTimesLastYear')


# - No clear trend in attrition with respect to number of trainings.

# In[ ]:


area_chart(df_eda,'absences_days')


# - No clear trend in attrition with respect to number of absence days.

# In[ ]:


area_chart(df_eda,'mean_actual_hours')


# - There is a clear trend, where the employees who work more hours, have the highest attrition

# In[ ]:


area_chart(df_eda,'YearsAtCompany')


# - Employees who stay at company for a longer period of time have lower attrition rate.

# In[ ]:


area_chart(df_eda,'YearsSinceLastPromotion')


# - No clear trend in attrition with respect to number of years since the last promotion.

# In[ ]:


area_chart(df_eda,'YearsWithCurrManager')


# - Employees who stay at company with the same manager for a longer period of time have lower attrition rate.

# In[ ]:


area_chart(df_eda,'DistanceFromHome')


# - No clear trend in attrition with respect to distance between home and work.

# #### Featuring Engineering

# - Label Encoding
# 
# Transform the categorical values of the relevant features into numerical ones using Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
df5=df4.copy()
# Create a list has all categorical features
cat_cols= df5.select_dtypes(include='object').columns
# Fitted SKlearn LabelEncoder
label_encoder = LabelEncoder()
# Dictionary mapping labels of the categorical features to their integer values 
label_encoder_data_dictionary={}
for col in cat_cols:
    # Encode labels in categorical column.
    df5[col]= label_encoder.fit_transform(df5[col])
    res = {}
    for cl in label_encoder.classes_:
        res.update({cl:label_encoder.transform([cl])[0]})
    label_encoder_data_dictionary[col] = res


# In[ ]:


# Check if the dataset has any categorical values
cat_cols= df5.select_dtypes(include='object').columns
print(cat_cols)


# In[ ]:


# Dictionary mapping labels of the categorical features to their integer values 
label_encoder_data_dictionary


# - Features scaling.
# 
# This dataset has many features computed by different measurements and units, so it is necessary to scale the features (except Id and SalePrice columns) to apply the ML models.
# 
# In this project, I use PowerTransformer normalization in order to handle the outliers better.

# In[ ]:


from sklearn.preprocessing import PowerTransformer
df6=df5.copy()
df6=df6.loc[:, ~df6.columns.isin(['Attrition','MonthlyIncome'])]
scalar=PowerTransformer()
df6=pd.DataFrame(scalar.fit_transform(df6),columns = df6.columns)
df6 = pd.concat([df6, df5.reset_index()[['Attrition','MonthlyIncome']]],axis=1,ignore_index=False)
df6.head()


# In[ ]:


# Apply log transformation on the MonthlyIncome feature (high skewness)
df6['MonthlyIncome'] = np.log(df6.MonthlyIncome)


# In[ ]:


#  Univariate analysis
df6.skew()


# In[ ]:


df6.plot(kind='box', subplots = True, layout = (5,6), figsize=(12,15));


# Check the target feature (Attrition)

# In[ ]:


df6['Attrition'].unique()


# - Features Selection
# To reduce the features for the linear regression and  skip the not useful ones.

# In[ ]:


# Plot a heatmap of correlation
corr = df6.corr()
plt.figure(figsize = (20,20))
sns.heatmap(corr,cbar=True)
plt.show()


# There is multi collinearity among some features

# In[ ]:


# # Select the unuseful independent feature from X-train and drop them.
# # Using Lasso Regression model and selectFromModel object, which will select the features with non-zero coefficients.
# from sklearn.linear_model import Lasso
# from sklearn.feature_selection import SelectFromModel
# # Select alpha=1.0% (equivalent of penalty).Bigger than this will select less features.
# sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
# sel_model.fit(X_train, y_train)
# # show the results, which True means that this feature is important to the Regression algorithm and false means not.
# for col in X_train.columns:
#     print(col,sel_model.get_support()[X_train.columns.get_loc(col)])


# In[ ]:


# Unstack and sort the DataFrame to get the most correlated pairs.
corr = df6.corr().abs()
unstack_corr = corr.unstack()
sorted_unstack_corr = unstack_corr.sort_values(kind="quicksort")


# In[ ]:


# Iterate over sorted_unstack_corr and find if it has collinearity more than 0.75.
for i in range (0,len(sorted_unstack_corr)):
    if  sorted_unstack_corr[i]>0.75 and sorted_unstack_corr[i]<1.0:
        print (sorted_unstack_corr[i],sorted_unstack_corr.index[i])


# 'YearsWithCurrManager'and 'YearsAtCompany' are highly coorelated, so drop 'YearsWithCurrManager'.
# 'PerformanceRating', 'PercentSalaryHike' are highly coorelated, so drop 'PerformanceRating'.

# In[ ]:


df7=df6.copy()
df7.drop(columns={'YearsWithCurrManager'},inplace=True)
df7.shape


# In[ ]:


# Capture the dependent feature from the dataset (y-train).
y_train=df7[['Attrition']]


# In[ ]:


# Capture the independent feature from the dataset (X-train).
X_train=df7.drop(['Attrition'],axis=1)


# #### Model building Phase
# 1. Split the data
# 2. Defining evaluation functions.
# 3. Machine Learning Models.
# 4. Model Comparison.

# Split the train data as x_train , x_test, y_train, y_test using sklearn library

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 3)


# - Defining evaluation functions.

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import zero_one_loss
from sklearn.metrics import confusion_matrix

def evaluation(y_test, y_pred):
    AccScore = accuracy_score(y_test, y_pred, normalize=False)
    print('Accuracy Score is : ', AccScore)
    ClassificationReport = classification_report(y_test,y_pred)
    print('Classification Report is:')
    print(ClassificationReport)
    ROCAUCScore = roc_auc_score(y_test,y_pred, average='micro') #it can be : macro,weighted,samples
    print('ROCAUC Score : ', ROCAUCScore)
    ZeroOneLossValue = zero_one_loss(y_test,y_pred,normalize=False)
    print('Zero One Loss Value : ', ZeroOneLossValue )
    ConfusionMatrix=confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', ConfusionMatrix )
#   Plot ROC Curve
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    area = auc(fpr, tpr)
    plt.figure(figsize=(18,5))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % area)
    plt.plot([0, 1], [0, 1],color='g', marker='_')
    plt.title('ROC Curve', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.legend(loc = 'lower right', fontsize = 12)
    plt.show()


# In[ ]:


# Define Cross_Validation
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


# - Machine Learning Classification Models.
# 
# Will use the most common types of classification algorithms: Logistic Regression, Naïve Bayes, Stochastic Gradient Descent, K-Nearest Neighbours, Decision Tree, Random Forest, Ridge, Bagging, Gradient Boosting, and Support Vector Machine

# In[ ]:


# Import classification models from sklearn library
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Dictionary contains all classification algorithms that we want to train and its hyperparameters that we want to tune.
clf_algos={
#         'LogisticRegression':{
#             'model':LogisticRegression(),
#             'params':{
# #                 'penalty' : ['l2'],
# #                 'tol': list(np.logspace(np.log(1e-5), np.log(1e-2), num = 10, base = 3)),
# # #               'C': list(np.linspace(0.1, 2.0, 20)),
# #                 'fit_intercept': [True, False],
#                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# # #                 ,
# # #               'max_iter': list(range(50, 501))
#             }
#         }
#         ,
#         'RidgeClassifier':{
#             'model':RidgeClassifier(),
#             'params':{
# #                 'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#             }
#         }
#         ,
#         'KNeighborsClassifier':{
#             'model':KNeighborsClassifier(),
#             'params':{
# #                 'n_neighbors' : range(1, 21, 2),
# #                 'weights' : ['uniform', 'distance'],
# #                 'metric' : ['euclidean', 'manhattan', 'minkowski']
#             }
#         },
#         'SVC':{
#             'model':SVC(),
#             'params':{
# # #                'C': list(np.linspace(0.1, 2.0, 10)),
# #                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
# #                  'degree': list(range(2, 6)),
# #                  'gamma': ['auto', 'scale'],
# # #                'tol': list(np.logspace(np.log(1e-5), np.log(1e-2), num = 10, base = 3)),
# # #                'max_iter': list(range(-1, 101))
#             }
#         },
#         'BaggingClassifier':{
#             'model':BaggingClassifier(),
#             'params':{
# # #                 'n_estimators' : [10, 100, 1000]
#             }
#         },
        'RandomForestClassifier':{
            'model':RandomForestClassifier(),
            'params':{
#                 'n_estimators' : [10,90, 100,115,130, 1000],
#                 'criterion': ['gini', 'entropy'],
#                  'max_features': ['sqrt','auto', 'log2', None]
            }
        }
#         ,
#         'GradientBoostingClassifier':{
#             'model':GradientBoostingClassifier(),
#             'params':{
# #                 'n_estimators' : [10, 100, 1000],
# #                 'learning_rate' : [0.001, 0.01, 0.1],
# #                 'subsample' : [0.5, 0.7, 1.0],
# #                 'max_depth' : [3, 7, 9]
#             }
#         }
#         ,
#         'SGDClassifier':{
#             'model':SGDClassifier(),
#             'params':{
# # #                 'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
# # # #               'n_iter': [1000], # number of epochs
# # #                 'loss': ['log','squared_loss'], # logistic regression,
# # #                 'penalty': ['l2'],
# # #                 'n_jobs': [-1],
# # #                 'learning_rate' : ['optimal']
#             }
#         }
#         ,
#         'DecisionTreeClassifier ':{
#             'model':DecisionTreeClassifier(),
#             'params':{
                
# #                  'criterion': ['gini', 'entropy'],
# #                  'splitter': ['best', 'random'],
# #                  'max_depth': list(range(3, 15,1)),
# #                  'min_samples_split': list(range(2, 15,1)),
# #                  'min_samples_leaf': list(range(1, 10,1)),
# #                  'max_features': ['sqrt','auto', 'log2', None]
#             }
#         }
#         ,
#         'GaussianNB ':{
#             'model':GaussianNB(),
#             'params':{
# # #                 'var_smoothing': np.logspace(0,-9, num=100)
#             }
#         }
}


# In[ ]:


# Import libraries to train, evaluate and save the models.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import joblib

# Function carry on multitasks:
# 1- Train different types of classification algorithms.
# 2- Hyperparameter tuning using GridSearchCV.
# 3- Evaluate each classification algorithm by diffirent types of metrices.
# 4- Make a report with best score and best Hyperparameter for each classification algorithm.
# 5- Save each trained model in pickle file using joblib.

def best_clf_model(X_train, y_train, X_test, y_test):
#     Lists to add the result in
    ml_models,model_scores,predictions=[],[],[]
#     Cross Validation
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
#     Loop on clf_algo dictionary
    for algo_name,config in clf_algos.items():
#         use GridSearchCV to tune the hyperparameteres for the algorithm
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False).fit(X_train, y_train)
#     Find y_predict
        y_pred=gs.predict(X_test)
#     Error validation functions
        print(f'{algo_name} Evaluation:')        
        evaluation(y_test, y_pred)
        cv_score = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5,n_jobs=-1,verbose=3).mean()
#         Add the scores to the lists
        model_scores.append({

                            "Model_name": algo_name,
                            "best_score":gs.best_score_,
                            "Cross_Validation_Score": cv_score,
                            "best_params":gs.best_params_,
                            "best_estimator":gs.best_estimator_})
        predictions.append({"Model_name": algo_name,"y_pred":y_pred})
        print(f'{algo_name} is done')
#         Save the trained model
        joblib.dump(gs.best_estimator_, f'{algo_name}.pkl', compress = 1)
    return pd.DataFrame(model_scores, columns=["Model_name","best_score",
                                            "Cross_Validation_Score",
                                            "best_params",
                                            "best_estimator"]
                       ).sort_values(by="Cross_Validation_Score"),pd.DataFrame(predictions, columns=["Model_name","y_pred"])


# In[ ]:


clf_model_scores,clf_predictions=best_clf_model(X_train, y_train, X_test,y_test)


# In[ ]:


# A report with best score and best Hyperparameter for each classification algorithm.
clf_model_scores


# After performing the data preprocessing, model building and validation, it's clear that RandomForestClassifier is the best model can be deployed to predict the probability that an employee will leave the company.

# Accuracy  score for the model is very well, which means the model performed extremely well. Also the other performance Metric's values were greater than 95%.
# 
# The model classified both of the classes of Attrition properly. High F1 score represents how much the model has learned to predict the employee Attrition properly, which confirmed by the Confusion Matrix.
# 
# From ROCAUC Score value (0.97), the model has good prediction performance.

# In[ ]:


#  Best parameters for the RandomForestClassifier model is:
# model_scores.best_estimator[5]


# In[ ]:


# Print the predictions for each model
clf_predictions


# ## Predict Years at Company

# #### Model building Phase
# 1. Prepare data
# 2. Defining evaluation functions.
# 3. Machine Learning Models.
# 4. Model Comparison.

# #### 1. Split the train data as x_train , x_test, y_train, y_test

# In[ ]:


#  Drop all data where Attrition is equal 0 (No Attrition)
df8=df7[df7.Attrition==1]
df8.head()


# In[ ]:


#  Drop Attrition feature because it has just one value, which won't be affected on the model training
df8=df8.loc[:, ~df8.columns.isin(['Attrition'])]


# In[ ]:


# Capture the dependent feature from the dataset (y-train).
y_train=df8[['YearsAtCompany']]


# In[ ]:


# Capture the independent feature from the dataset (X-train).
X_train=df8.drop(['YearsAtCompany'],axis=1)


# In[ ]:


# Split the train data as x_train , x_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 3)


# - Machine Learning Regression Models.
# 
# Will use the most common types of regression algorithms: Linear Regression, Lasso, Ridge Regression, Decision Tree Regressor, ElasticNet, Random Forest Regression, Gradient Boosting Regression, XGBoost Regressor, and Support Vector Machine

# In[ ]:


get_ipython().system('pip install xgboost')


# In[ ]:


# Import classification models from sklearn library
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.tree import DecisionTreeRegressor

# Dictionary contains all regression algorithms that we want to train and its hyperparameters that we want to tune.
reg_algos={
#         'linear_regression':{
#             'model':LinearRegression(),
#             'params':{
#                 'normalize':[True,False]
#             }
#         },
#         'lasso':{
#             'model':Lasso(),
#             'params':{
#                 'alpha':[1,2],
#                 'selection':['random','cycle']
#             }
#         },
#         'ridge_regression':{
#             'model':Ridge(),
#             'params':{}
#         },
#         'DecisionTreeRegressor':{
#             'model':DecisionTreeRegressor(),
#             'params':{
#                 'criterion':['mse','friedman_mse'],
#                 'splitter':['best','random'],
#                 'max_depth' : [10],
#                 'min_samples_leaf' : [2]
#             }
#         },
#         'elastic_net':{
#             'model':ElasticNet(),
#             'params':{}
#         },
#         'SVR':{
#             'model':SVR(),
#             'params':{
#                 'C':[100000,0.7],
#                 'kernel' : ['rbf'],
#                 'gamma' : ['auto'],
#                 'degree':[4],
# #                 'epsilon':[0.002],
#                 'coef0':[20]
#             }
#         },
#         'Random_Forest_Regressor':{
#             'model':RandomForestRegressor(),
#             'params':{
#                 'n_estimators':[100,1500],
#                 'max_depth' : [3]
#             }
#         },
        'GradientBoostingRegressor':{
            'model':GradientBoostingRegressor(),
            'params':{
                'n_estimators':[100,1500],
                'learning_rate':[0.01,0.015],
                'max_depth':[3],
                'min_samples_leaf':[1],
                'random_state':[2],
                'subsample' : [0.2]
            }
        }
#         ,
#         'XGBoost Regressor':{
#             'model':XGBRegressor(),
#             'params':{
#                 'n_estimators':[100],
#                 'learning_rate':[0.01]
#             }
#         }

}


# In[ ]:


# Import libraries to train, evaluate and save the models.
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

# Function carry on multitasks:
# 1- Train different types of Regression algorithms.
# 2- Hyperparameter tuning using GridSearchCV.
# 3- Evaluate each regression algorithm by diffirent types of metrices.
# 4- Make a report with best score and best Hyperparameter for each regression algorithm.
# 5- Save each trained model in pickle file using joblib.

def best_reg_model(X_train, y_train, X_test, y_test):
#     Lists to add the result in
    ml_models,model_scores,predictions=[],[],[]
#    Cross Validation
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
#     Loop on reg_algo dictionary
    for algo_name,config in reg_algos.items():
#         use GridSearchCV to tune the hyperparameteres for the algorithm
        gs=GridSearchCV(config['model'],config['params'],cv=cv,return_train_score=False).fit(X_train, y_train)
#     Find y_predict
        y_pred=gs.predict(X_test)
#     Error validation functions
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r_squared = r2_score(y_test, y_pred)
        rmse_cv = np.sqrt(-cross_val_score(gs, X_train, y_train, scoring="neg_mean_squared_error", cv=5)).mean()
#         Add the scores to the lists
        model_scores.append({
                          "Model_name": algo_name,
                          "best_score":gs.best_score_,
                          "mean_absolute_error": mae,
                          "mean_squared_error": mse,
                          "root_mean_squared_error": rmse,
                          "r2_score": r_squared,
                          "RMSE_Cross_Validation": rmse_cv,
                          "best_params":gs.best_params_,
                            "best_estimator":gs.best_estimator_})
        predictions.append({"Model_name": algo_name,"y_pred":y_pred})
        joblib.dump(gs.best_estimator_, f'{algo_name}.pkl', compress = 1)
#         Save the trained model
        print(f'{algo_name} is done')        
    return pd.DataFrame(model_scores, columns=["Model_name","best_score",
                                            "mean_absolute_error",
                                            "mean_squared_error", 
                                            "root_mean_squared_error",
                                            "r2_score",
                                            "RMSE_Cross_Validation",
                                            "best_params",
                                            "best_estimator"]
                       ).sort_values(by="RMSE_Cross_Validation"
                                    ),pd.DataFrame(predictions, columns=["Model_name","y_pred"])


# In[ ]:


reg_model_scores,reg_predictions=best_reg_model(X_train, y_train, X_test,y_test)


# In[ ]:


# A report with best score and best Hyperparameter for each classification algorithm.
reg_model_scores


# In[ ]:


#  Stacked-bar chart show the r2_score and RMSE_Cross_Validation for each trained model
fig, ax = plt.subplots()
ax.bar(reg_model_scores["Model_name"], reg_model_scores["r2_score"], width = 0.35 , label='r2_score')
ax.bar(reg_model_scores["Model_name"], reg_model_scores["RMSE_Cross_Validation"],width = 0.35 , label='RMSE_Cross_Validation')
ax.set_ylabel('Scores')
plt.title("Evaluation of Models Based on RMSE (Cross-Validated)")
plt.xticks(rotation=90)
ax.legend()
plt.show()


# After performing the data preprocessing, model building and validation, it's clear that GradientBoostingRegressor is the best model can be deployed to predict how many years may an employee stay at the company before leaving.
# 
# Accuracy score for the model is very well, which means the model performed extremely well. Also the other performance Metric's values were greater than 90%.
# 
# It's better for RMSE Cross Validation score to be less than 25%, but r2 score is ver high. The model has good prediction performance.

# In[ ]:


# Dictionary mapping labels provided by data source 
Provided_data_dictionary={
        'Education':{
            'Below College':1,
            'College':2,
            'Bachelor':3,
            'Master':4,
            'Doctor':5
        }
        ,
        'EnvironmentSatisfaction':{
            'Low':1,
            'Medium':2,
            'High':3,
            'Very High':4
        }
        ,
        'JobInvolvement':{
            'Low':1,
            'Medium':2,
            'High':3,
            'Very High':4
        }
        ,
        'JobSatisfaction':{
            'Low':1,
            'Medium':2,
            'High':3,
            'Very High':4
        }
        ,
        'PerformanceRating':{
            'Low':1,
            'Good':2,
            'Excellent':3,
            'Outstanding':4
        }
        ,
        'RelationshipSatisfaction':{
            'Low':1,
            'Medium':2,
            'High':3,
            'Very High':4
        }
        ,
        'WorkLifeBalance':{
            'Bad':1,
            'Good':2,
            'Better':3,
            'Best':4
        }
    }
# Dictionary mapping labels After Label Encoding process and the provided from data source
label_encoder_data_dictionary.update(Provided_data_dictionary)
data_dictionary=label_encoder_data_dictionary.copy()


# In[ ]:


data_dictionary


# Load the models

# In[ ]:


Attrition_clf=joblib.load("RandomForestClassifier.pkl")
years_at_company_reg=joblib.load("GradientBoostingRegressor.pkl")


# In[ ]:


# Attrition classification function with predict the years at the company if the attrition is yes
def Attrition_classification(
#     Age of the employee (18-60)
    Age,
#     'Non-Travel','Travel_Frequently','Travel_Rarely'
    BusinessTravel,
#     'Human Resources', 'Research & Development', 'Sales'
    Department,
#     Distance from home in kms
    DistanceFromHome,
#     'Below College','College','Bachelor','Master','Doctor'
    Education,
#     'Human Resources','Life Sciences','Marketing','Medical','Other','Technical Degree'
    EducationField,
#     'Female', 'Male'
    Gender,
#     A scale of 1 to 5
    JobLevel,
#     'Healthcare Representative','Human Resources','Laboratory Technician','Manager',
#     'Manufacturing Director','Research Director','Research Scientist','Sales Executive',
#     'Sales Representative'
    JobRole,
#     'Divorced', 'Married', 'Single'
    MaritalStatus,
#     Total number of companies the employee has worked for
    NumCompaniesWorked,
#     Percent salary hike for last year
    PercentSalaryHike,
#     Stock option level of the employee
    StockOptionLevel,
#     Total number of years the employee has worked so far
    TotalWorkingYears,
#     Number of times training was conducted for this employee last year
    TrainingTimesLastYear,
#     Total number of years spent at the company by the employee
    YearsAtCompany,
#     Number of years since last promotion
    YearsSinceLastPromotion,
#     Number of days that the employee didn't show up in 2015
    absences_days,
#     Mean of actual of work-hours in 2015
    mean_actual_hours,
#     'Low', 'Medium', 'High', 'Very High'
    JobInvolvement,
#     'Low', 'Good', 'Excellent', 'Outstanding'
    PerformanceRating,
#     'Low', 'Medium', 'High', 'Very High'
    EnvironmentSatisfaction,
#     'Low', 'Medium', 'High', 'Very High'
    JobSatisfaction,
#     'Bad', 'Good', 'Better', 'Best'
    WorkLifeBalance,
#     Monthly income in rupees per month
    MonthlyIncome
    ):
    attrs=np.zeros(len(X_train.columns)+1)
    attrs[0]=Age
    attrs[1]=data_dictionary['BusinessTravel'][BusinessTravel]
    attrs[2]=data_dictionary['Department'][Department]
    attrs[3]=DistanceFromHome
    attrs[4]=data_dictionary['Education'][Education]
    attrs[5]=data_dictionary['EducationField'][EducationField]
    attrs[6]=data_dictionary['Gender'][Gender]
    attrs[7]=JobLevel
    attrs[8]=data_dictionary['JobRole'][JobRole]
    attrs[9]=data_dictionary['MaritalStatus'][MaritalStatus]
    attrs[10]=NumCompaniesWorked
    attrs[11]=PercentSalaryHike
    attrs[12]=StockOptionLevel
    attrs[13]=TotalWorkingYears
    attrs[14]=TrainingTimesLastYear
    attrs[15]=YearsAtCompany
    attrs[16]=YearsSinceLastPromotion
    attrs[17]=absences_days
    attrs[18]=mean_actual_hours
    attrs[19]=data_dictionary['JobInvolvement'][JobInvolvement]
    attrs[20]=data_dictionary['PerformanceRating'][PerformanceRating]
    attrs[21]=data_dictionary['EnvironmentSatisfaction'][EnvironmentSatisfaction]
    attrs[22]=data_dictionary['JobSatisfaction'][JobSatisfaction]
    attrs[23]=data_dictionary['WorkLifeBalance'][WorkLifeBalance]
    attrs[24]=MonthlyIncome
#     Classification prediction for ATTRITION
    result=Attrition_clf.predict([attrs])[0]
    if result==1:
#         Drop YearsAtCompany feature from the arry because it's the target 
        attrs2=np.delete(attrs, obj=15)
#     Prediction for number of years at the company
        years_at_company=years_at_company_reg.predict([attrs2])[0]
        return f"There is a high probability that the employee will LEAVE the company in {years_at_company} years"
    return "There is a high probability that the employee will STAY at the company"


# In[ ]:


# Use Attrition_classification on a new employee
Attrition_classification(
#     Age of the employee (18-60)
    25,
#     'Non-Travel','Travel_Frequently','Travel_Rarely'
    'Travel_Rarely',
#     'Human Resources', 'Research & Development', 'Sales'
    'Sales',
#     Distance from home in kms
    30,
#     'Below College','College','Bachelor','Master','Doctor'
    'College',
#     'Human Resources','Life Sciences','Marketing','Medical','Other','Technical Degree'
    'Marketing',
#     'Female', 'Male'
    'Male',
#     A scale of 1 to 5
    3,
#     'Healthcare Representative','Human Resources','Laboratory Technician','Manager',
#     'Manufacturing Director','Research Director','Research Scientist','Sales Executive',
#     'Sales Representative'
    'Manager',
#     'Divorced', 'Married', 'Single'
    'Married',
#     Total number of companies the employee has worked for
    3,
#     Percent salary hike for last year
    12,
#     Stock option level of the employee, A scale of 0 to 3
    2,
#     Total number of years the employee has worked so far
    6,
#     Number of times training was conducted for this employee last year
    1,
#     Total number of years spent at the company by the employee
    3,
#     Number of years since last promotion
    1,
#     Number of days that the employee didn't show up in 2015
    6,
#     Mean of actual  of work-hours in 2015
    7,
#     'Low', 'Medium', 'High', 'Very High'
    'Medium',
#     'Low', 'Good', 'Excellent', 'Outstanding'
    'Good',
#     'Low', 'Medium', 'High', 'Very High'
    'High',
#     'Low', 'Medium', 'High', 'Very High'
    'Very High',
#     'Bad', 'Good', 'Better', 'Best'
    'Good',
#     Monthly income in rupees per month
    30000
    )


# ### Conclusion
# The patterns in data can help the management understand why the employee attrition is high, and what can be done to lower it.
# 
# During EDA, a lot of insights about Employee Attrition discovered. Some of the main reasons that cause employee attrition to rise is the HR work, bad work-life balance, Frequent Travels and Single Employees.
# 
# This model can help the management reduce costs related to hiring  new employees.
