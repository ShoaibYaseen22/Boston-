#!/usr/bin/env python
# coding: utf-8

# # Delivered To:

#                  ::: Dr Ayesha Hakim :::

# In[1]:


import numpy as np
import pandas as pd


# # Import the dataset

# In[2]:


df= pd.read_csv('Boston Dataset.csv')


# In[3]:


df.head(50)


# In[4]:


df.tail(50)


# # Shape of dataset

# In[5]:


print('Shape of Training dataset:', df.shape)


# # Checking null values for training dataset

# In[6]:


df.isnull().sum()


# # The Target Variable is the last one which is called MEDV.

# ## Here lets change ‘medv’ column name to ‘Price’

# In[7]:


df.rename(columns={'MEDV':'PRICE'}, inplace=True)
df


# # Exploratory Data Analysis

# ## Information about the dataset features

# In[8]:


df.info()


# ## Describe

# In[9]:


df.describe()


# ## Minimum price of the data

# In[10]:


minimum_price = np.amin(df['PRICE'])


# ## Maximum price of the data

# In[11]:


maximum_price = np.amax(df['PRICE'])


# ## Mean price of the data

# In[12]:


mean_price = np.mean(df['PRICE'])


# ## Median price of the data

# In[13]:


median_price = np.median(df['PRICE'])


# ## Standard deviation of prices of the data

# In[14]:


std_price = np.std(df['PRICE'])


# # Show the calculated statistics

# In[15]:


print("Statistics for Boston housing dataset:\n")
print("Minimum PRICES: ${}".format(minimum_price)) 
print("Maximum PRICES: ${}".format(maximum_price))
print("Mean PRICES: ${}".format(mean_price))
print("Median PRICES ${}".format(median_price))
print("Standard deviation of PRICES: ${}".format(std_price))


# # Feature Observation

# ## Finding out the correlation between the features

# In[16]:


corr = df.corr()
corr.shape


# ## Plotting the heatmap of correlation between features

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar=False, square= True, fmt='.2%', annot=True, cmap='Greens')


#                                                  ::: HeatMap :::

# # Checking the null values using heatmap

# ## There is any null values are occupyed here

# In[19]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ### Note: There are no null or missing values here.

# In[20]:


sns.set_style('whitegrid')
sns.countplot(x='RAD',data=df)


#                           ::: Counting For RAD Values :::

# In[21]:


sns.set_style('whitegrid')
sns.countplot(x='CHAS',data=df)


#                          ::: Counting For CHAS Feature :::

# In[22]:


sns.set_style('whitegrid')
sns.countplot(x='CHAS',hue='RAD',data=df,palette='RdBu_r')


#                          ::: CHAS DATA :::

# In[23]:


sns.histplot(data=df, x='AGE', color='darkred', bins=40)


#                          ::: HOUSE'S AGE Features Understanding :::

# In[24]:


sns.histplot(df['CRIM'].dropna(), kde=False, color='darkorange', bins=40)


#                          ::: CRIM RATE :::

# In[25]:


sns.histplot(df['RM'].dropna(), color='darkblue', bins=40)


#                          ::: Understanding Number of ROOMS into the HOUSES :::

# # Feature Selection

# ## Lets try to understand which are important feature for this dataset

# In[26]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# ## Independent Columns

# In[27]:


X = df.iloc[:,0:13]


# ## Target Column i.e PRICE range

# In[28]:


y = df.iloc[:,-1]


# In[29]:


y = np.round(df['PRICE'])


# # Apply SelectKBest class to extract top 5 best features

# In[30]:


bestfeatures = SelectKBest(score_func=chi2, k=5)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# ## Concat two dataframes for better visualization

# In[31]:


featureScores = pd.concat([dfcolumns,dfscores],axis=1)


# ## Naming the dataframe Columns

# In[32]:


featureScores.columns = ['Specs','Score'] 
featureScores


# # Print 5 best features

# In[33]:


print(featureScores.nlargest(5,'Score'))


# # Feature Importance

# In[34]:


from sklearn.ensemble import ExtraTreesClassifier


# In[35]:


model = ExtraTreesClassifier()
model.fit(X,y)


# ## Use inbuilt class feature_importances of tree based classifiers

# In[36]:


print(model.feature_importances_) 


# ## Plot graph of feature importances for better visualization

# In[37]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


#         ::: Important Features rated by target variable correlation :::

# # Model Fitting
# 
# 
# # Linear Regression

# ### Value Assigning

# In[38]:


x=df.iloc[:,0:13]
y=df.iloc[:,-1]


# In[39]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[40]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[41]:


y_pred=model.predict(x_train)


# In[42]:


print("Training Accuracy:",model.score(x_train,y_train)*100)


# In[43]:


print("Testing Accuracy:",model.score(x_test,y_test)*100)


# In[44]:


from sklearn.metrics import mean_squared_error, r2_score


# In[45]:


print("Model Accuracy:",r2_score(y,model.predict(x))*100)


# In[46]:


plt.scatter(y_train,y_pred)
plt.xlabel('PRICES')
plt.ylabel('PREDICTED PRICES')
plt.title('PRICES VS RPEDICTED PRICES')
plt.show()


#             :::  See! how data points are predicted :::

# # Checking Residuals

# In[47]:


plt.scatter(y_pred,y_train-y_pred)
plt.title('RPEDICTED VS RESIDUALS')
plt.xlabel('PREDICTED')
plt.ylabel('RESIDUALS ')
plt.show()


#                    ::: Predicted Vs Residuals :::
# 

# # Checking Normality of of Errors

# In[48]:


sns.histplot(y_train-y_pred)
plt.title('Histogram of RESIDUALS')
plt.xlabel('RESIDUALS')
plt.ylabel('FREQUENCY ')
plt.show()


#                     ::: Hist Plotting for residuals :::

# # Random Forest Regression

# In[49]:


x=df.iloc[:,[-1,5,10,4,9]]
y=df.iloc[:,[-1]]


# In[50]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[51]:


from sklearn.ensemble import RandomForestRegressor
reg=RandomForestRegressor()
reg.fit(x_train,y_train)


# In[52]:


y_pred = reg.predict(x_train)


# In[53]:


print("Training Accuracy:",reg.score(x_train,y_train)*100)


# In[54]:


print("Testing Accuracy:",reg.score(x_test,y_test)*100)


# # Visualizing the difference between actual PRICES and PREDICTED values

# In[55]:


plt.scatter(y_train,y_pred)
plt.xlabel('PRICES')
plt.ylabel('PREDICTED PRICES')
plt.title('PRICES VS PREDICTED PRICES')
plt.show()


#                    ::: Linear Regression plotting data points :::

# # Prediction and Final Score:

# ### Finally we made it!!!

# # 1.Linear Regression

# ## Training Accuracy: 77.30135569264233
# 
# ## Testing Accuracy: 58.9222384918251
# 
# ## Model Accuracy: 73.73440319905033

# # 2. Random Forest Regressor
# 
# 

# 
# ## Training Accuracy: 99.99323673544639
# 
# ## Training Accuracy: 99.99323673544639
# 

# # Delivered By:

#                           ::: M Yasir Madni :::
#                           ::: Shoaib Yaseen :::
#                           ::: Muhammad Riyan:::
#                           ::: Hassan Raza   :::
#                           ::: Raza Abbas    :::

# In[ ]:




