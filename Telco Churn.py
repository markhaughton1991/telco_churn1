#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# # Telco Churn 

# ### Table of Contents
# #### 1. Introduction
# #### 2. Data Exploration
# #### 3. Data Cleaning
# #### 4. Data Analysis and visualisation
# #### 5. Predictive Analytics

# ## 1. Introduction
# 
# #### This is a project to determine customer churn for a telco company.  The dataset used is a dummy test dataset available on Kaggle: https://www.kaggle.com/blastchar/telco-customer-churn
# 
# 
# 
# #### The aim of this analysis is to identify the cause of customer churn within a large Telco. The dataset of roughly 7,000 customers is a subset of data which is thought to be representative of the whole customer base. 
# 
# #### The insight gained from this analysis is intended to provide the senior leadership with factual information on which to base business and marketing decisions. 
# 

# ## 2. Data Exploration 
# 
# #### We now want to familiarise ourselves with the dataset. This will allow us to understand what data might be important for our churn analysis, and also give us a feel for the quality of data. Poor quality data will require significant cleaning, which may also reduce the accuracy of results.

# The following code is used to ingest the dataset.
# Watson Studio has a quick and easy way to pull this in without writing a single line of code.
# It is also setup to return the first 5 rows of the dataset

# In[1]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3
import numpy as np # library for linear algebra
import seaborn as sns # library for graphs
import matplotlib.ticker as mtick 
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff

init_notebook_mode(connected=True)

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
client_cbd7f294ab5c4bc0a708d8b14dc5910e = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='s8fbIREqfzPvw9hDQ1RfhJZRUsPmTR-rIpObVkAbKuAP',
    ibm_auth_endpoint="https://iam.eu-gb.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_cbd7f294ab5c4bc0a708d8b14dc5910e.get_object(Bucket='telcochurn-donotdelete-pr-udxymttp3w9mxo',Key='WA_Fn-UseC_-Telco-Customer-Churn.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

customers = pd.read_csv(body) #auto generated name


# In[2]:


customers.head()


# Identify the data types of each column

# In[3]:


customers.dtypes


# Most of the above seems to make sense. Could change the following though:
# 
# SeniorCitizen is listed a int64, so should investigate if it uses zeroes and ones rather than Yes and No
# 
# TotalCharges is stored as an object, so should investiagate if this should change to a number

# Search to see if there are any 1's

# In[4]:


customers.apply(pd.value_counts)


# We can see that SeniorCitizen is either 0 or 1, but "Partner" is stored as an object, but this should also be a boolean value of zero or one. 

# ## 3. Data Cleaning
# 
# #### Prior to analysing the data it is imporant to format the data and remove inconsistencies and incorrect values which may alter results. There are several techniques which can be used to clean the data. 

# When dealing with boolean values, it is usuful to replace character values with numerical values. This will allow for easy comparison with other variables and simplified tabulation

# In[5]:


customers['Partner'].replace(to_replace= 'Yes', value=1, inplace = True)
customers['Partner'].replace(to_replace= 'No', value=0, inplace = True)


# In[6]:


customers['Partner'].head()


# Note:
# I initially used the above logic to replace every "Yes" and "No" in the dataset with either a 1 or a zero. However, this returned multiple incorrect values as some Customer IDs had "YES" and "NO" within their customer ID. 
# 
# This was only found out after making the change and then looking at the head. 
# 
# So, I should remove all columns from the dataframe which may have "No" or "Yes" which I don't want changed

# Somethinng seems to be going on with the TotalCharges column too.
# 
# When I apply the above logic to all columns I get the following:
# 
# ![image.png](attachment:image.png)

# Total Charges does not have any "Yes" or "No" values, but it is listed as an object, so I should change this to a number

# In[7]:


customers.TotalCharges = pd.to_numeric(customers.TotalCharges, errors='coerce') #errors = 'coerce' results in erroneous values produce "NaN"

# customers["TotalCharges"] = customers.TotalCharges.astype(float) This will only change a string to a float. Will not change object to float


# In[8]:


df2 = customers.iloc[:,1:] #to remove customer ID from dataframe
df_binary = pd.get_dummies(df2) #applying the logic to replace Yes and No values with 1's and 0's
df_binary.head()


# In[9]:


df_binary.columns.values #to check if all of the columns in the dataframe are sensible 


# #### We want to check if there are any null values which may skew our results. We can quickly see this by counting the number of null values 

# In[10]:


df_binary.isnull().sum() #check if any values are NaN


# We can see that there are 11 TotalCharges values with NaN values. We should remove these

# In[11]:


df_binary.dropna(inplace = True)


# In[12]:


df_binary.isnull().sum()


# ## 4. Data Analysis
# 
# #### Following data cleaning, we can now start to look at the data and try to gather some insight to inform business decisions.
# #### The most straight forward way to analyse the data is usually to visualise the data.
# #### In this scenario, we want to see what variables are correlated to churn. We can quite easily do this by using the .corr function

# In[13]:


#Now to determine the corrolation between Churn and the other columns
plt.figure(figsize=(15,8))
df_binary.corr()['Churn_Yes'].sort_values(ascending = False).plot(kind='bar').set_facecolor('xkcd:silver')


# The above shows that correlation is most significant with month to month contracts, absence of online security and absence tech support 
# 
# Tenure, two year contracts and no internet service are shown to have be negatively related to churn

# Lets look at the month to month contract

# In[14]:


customers.head()


# In[15]:


customers["Contract"].head()


# #### Visualise number of customers on different types of contracts
# 
# #### The following graph is from the plot.ly library. Watson Studio has the capability to import these libraries. This library includes interactive graphs which are useful for further interogatting data

# In[16]:


y_contract_type = customers['Contract'].value_counts().values # get count of contract types
x_contract_type = list(customers['Contract'].value_counts().index) # get corresponding labels
iplot([go.Bar(x=x_contract_type,y=y_contract_type)]) # plot them


# #### Now to find the tenure of customers for each of the above contract types

# In[17]:


#change this graph is possible
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharey = True, figsize = (20,6))

#tenure for month-to-month contracts
ax = sns.distplot(customers[customers['Contract']=='Month-to-month']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'turquoise',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax1)
ax.set_ylabel('# of Customers')
ax.set_xlabel('Tenure (months)')
ax.set_title('Month to Month Contract')

#tenure for one year contracts
ax = sns.distplot(customers[customers['Contract']=='One year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'steelblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax2)
ax.set_xlabel('Tenure (months)',size = 14)
ax.set_title('One Year Contract',size = 14)

#tenure for two year contracts
ax = sns.distplot(customers[customers['Contract']=='Two year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'darkblue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax3)

ax.set_xlabel('Tenure (months)')
ax.set_title('Two Year Contract')


# we can see from the above that its best for the Telco to get customers on 2 year contracts

# Now I want to analyse the types of services which customers are provided

# In[18]:


customers.columns.values


# #### Visualise services offered to customers 

# In[19]:


services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']


# In[20]:


fig, axes = plt.subplots(nrows = 3,ncols = 3,figsize = (15,12))
for i, item in enumerate(services):
    if i < 3:
        ax = customers[item].value_counts().plot(kind = 'bar',ax=axes[i,0],rot = 0)
        
    elif i >=3 and i < 6:
        ax = customers[item].value_counts().plot(kind = 'bar',ax=axes[i-3,1],rot = 0)
        
    elif i < 9:
        ax = customers[item].value_counts().plot(kind = 'bar',ax=axes[i-6,2],rot = 0)
    ax.set_title(item)


# #### Matrix comparing some variables which have high correlation with Churn 

# In[21]:


fig = ff.create_scatterplotmatrix(customers.loc[:1000, ['tenure', 'MonthlyCharges', 'TotalCharges']], diag='histogram',
                                  height=1000, width=1000)
iplot(fig, filename='Histograms along Diagonal Subplots')


# ### Churn Analysis
# Now I will analyse the Churn variable

# In[22]:


colors = ['black','blue']
ax = (customers['Churn'].value_counts()*100.0 /len(customers)).plot(kind='bar',stacked = True, rot = 0, color = colors,figsize = (8,6))
                                                                         
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('% Customers',size = 15)
ax.set_xlabel('Churn',size = 15)
ax.set_title('Churn Rate', size = 15)

# create a list to collect the plt.patches data
totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_width())

# set individual bar lables using above list
total = sum(totals)

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_x()+.15, i.get_height()-10.0,             str(round((i.get_height()/total), 1))+'%',
            fontsize=15,
            color='white',
           
           size = 15)


# Customer churn of 26.5% seems quite large for less than 6 years of data. Will need to look into this further

# ## Indepent variables and Churn

# ### Contract length and Churn

# Churn vs Contract type seems like a good place to start as we saw above that customers can have either month-to-month, one year or two year contracts. My assumption is that customers with month-to-month contracts will have higher churn rates than customers with 1 year and 2 year contracts as they can change service provider more often.

# In[23]:



contract_churn = customers.groupby(['Contract','Churn']).size().unstack()
ax = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar', #The 'T' is for "transpose". This 
                                                                width = 0.3,
                                                                rot = 0, 
                                                                figsize = (7,5),
                                                                color = colors)

ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('Percentage of Customers',size = 14)
ax.set_title('Contract Length vs Churn',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'red',
               weight = 'bold',
               size = 14)


# From the above, it is quite clear to see that the churn rate is lower for customers on longer contracts 

# ### Customers with dependents and Churn

# Maybe its possible that customers with dependents (children) have a lower churn rate as maybe it is more time consuming to move multiple MPNs to new service provider? Or maybe there is higher churn as other service providers may have better family plans? 

# In[24]:



dependent_churn = customers.groupby(['Dependents','Churn']).size().unstack()
ax = (dependent_churn.T*100.0 / dependent_churn.T.sum()).T.plot(kind='bar', #The 'T' is for "transpose". This 
                                                                width = 0.3,
                                                                rot = 0, 
                                                                figsize = (7,5),
                                                                color = colors)

ax.legend(loc='best',prop={'size':14},title = 'Churn')
ax.set_ylabel('Percentage of Customers',size = 14)
ax.set_title('Churn vs Dependents',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'red',
               weight = 'bold',
               size = 14)


# So that's quite interesting(ish). Customers with dependents have a lower churn rate. 

# ### Value of monthly charges and Churn 
# Another interesting area would be churn vs monthly charges

# In[25]:


ax = sns.distplot(customers.MonthlyCharges[(customers["Churn"] == 'No') ],
                color="black")
ax = sns.distplot(customers.MonthlyCharges[(customers["Churn"] == 'Yes') ],
                ax =ax, color="blue")


ax.legend(["Does not churn","Churn"])
ax.set_ylabel('Density')
ax.set_xlabel('Monthly Charges')
ax.set_title('Distribution of monthly charges by churn')


# Above shows that cuztomers with higher monthly charges are more likely to churn 

# 

# ## 5. Predictive Analytics

# #### This section will identify the likelyhood that a customer will churn based on the variables which we have so far analysed 

# In[26]:


# First thing I need to do is something which I should have done earlier...some more data cleaning.
# To make predicting churn a little more straight forward, change the value of churn to either 1 or 0, rather than Yes or No 

# df2['Churn'].replace(to_replace='Yes', value=1, inplace=True) # df2 is the original dataframe used above
# df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

customers.dropna(inplace = True)

df2 = customers.iloc[:,1:] # customers is the imported data
#Convertin the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Let's convert all the categorical variables into dummy variables
df_pred = pd.get_dummies(df2)


# In[27]:


y = df_pred['Churn'].values
X = df_pred.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features # adding back the proper column names


# In[28]:


from sklearn import metrics
from sklearn.model_selection import train_test_split


# ### The below value shows the accuracy of using Random Forests

# In[29]:


from sklearn.ensemble import RandomForestClassifier #imports the random forests decision tree prediction library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)


# In[30]:


print("Accuracy Score: " + str(metrics.accuracy_score(y_test, prediction_test)))
print("F1 score: \t" + str(metrics.f1_score(y_test, prediction_test, average='macro')))
print(metrics.classification_report(y_test, prediction_test))


# In[31]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weight_importances = weights.sort_values(ascending = False)[:10]

y_weight_importances = weight_importances.values 
x_weight_importances = list(weight_importances.index)
iplot([go.Bar(x=x_weight_importances,y=y_weight_importances)])


# ### This shows that customers with monthly contracts, shorter tenure and high chargers are most likely to churn... This is reasonable given the earlier analysis

# ### Will now perform logistic regression

# In[32]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# #### Shows it is similarly accurate to random forests

# In[33]:


from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))


# In[34]:


# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
weight_importances = weights.sort_values(ascending = False)[:10]

y_weight_importances = weight_importances.values 
x_weight_importances = list(weight_importances.index)
iplot([go.Bar(x=x_weight_importances,y=y_weight_importances)])


# #### This shows very similar results to the Random Forests. However, Internet service seems to show a larger importance than tenure

# #### With this knowledge, the business can act to reduce churn by offering new offers and incentives in these areas.

# ## 6. Results

# #### By comparing the predictive analysis of Random Fruits and Logistic Regression we see that Total Charges, Monthly Contracts, No Fiber Optic Internet Service Provided and No Online Security are areas with the highest churn. 
# 
# #### From the Logistic Regression analysis we can also see that new joiners who are Senior Citizens are not likely to churn. We can also see that Streaming of movies, paperless billing and streaming of TV arew also imporant factors as to whether or not a customer will churn.
# 

# ## 7. Conclusion
# 
# #### Based on the results, we can conclude the following:
# 1. Customers on short term contracts are more likely to churn. As a result, the business should consider offering incentives for customers to take up longer term contracts, or increased marketing for long term contracts.
# 2. Senior Citizens are less likely to churn. Based on this, the business should consider increased marketing towards senior citizens. This could include demographic specific adverts and marketing channels (radio, newspaper etc.)
# 3. Online security and tech support is deemed imporant, therefore this could be offered at a reduced rate.  
# 4. Streaming of movies and tv has been deemed imporant. The business could increase marketing for these features and also market these features to existing customers.
# 
# #### However, it is important to understand that correlation does not imply causation. For example, customers with dependents have been shown to be less likely to churn. This does not necessarily mean that we should try and market specifically towards customers with dependents. Rather, this may mean that a certain age group of the population is more likely to not churn. This would need to be looked into further

# In[ ]:




