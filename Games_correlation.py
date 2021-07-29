#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = [15,10] #Adjusts the configuration of the plots we will create
pd.set_option('display.max_rows', 20)


# In[2]:


#load the data
data = pd.read_csv('video_games_sales.csv')


# In[3]:


data.head()


# In[4]:


#Finding out mising data if there are ny

for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[5]:


#exploring data
data.dtypes


# In[6]:


# Creating correct year column
data['year'] = data['Year_of_Release'].astype('str').str[:4]
data['Critic_Score'] = data['Critic_Score'].astype('float64')


# In[7]:


data['Critic_Score'] = data['Critic_Score']/10


# In[ ]:





# In[8]:


data.head()


# In[ ]:





# In[9]:


#Scatter plot critic_score vs user_score

plt.scatter(x=data['Critic_Score'], y = data['User_Score'],alpha=0.1)
plt.title('Critic_score vs User_score')

plt.xlabel('Critic_Score')

plt.ylabel('User_Score')

plt.show()

#with alpha parameter we can observe data better


# In[ ]:





# In[10]:


sns.regplot(x='Global_Sales', y='User_Score', data=data, scatter_kws={'color':'red'}, line_kws={'color':'blue'})


# In pandas correlation matrix have 3 method, since we are going to use Linear Regression on numerical values default version of `corr` function(pearson) will be enough for us
# 
# Click on the box to learn more about other methods!
# <table align="left">
#   <td>
#     <a target="_blank" href="https://realpython.com/numpy-scipy-pandas-correlation-python/"><img width="35"src="https://files.realpython.com/media/real-python-logo-square.28474fda9228.png" /></a>
#   </td>
# </table>

# In[11]:


data.corr(method='pearson')


# In[12]:


corr_matrix = data.corr() 
sns.heatmap(corr_matrix, annot=True)

plt.title('Correlation Matrix for numerical values')

plt.show()


# # Correlation for Categorical values 
# 
# Since corr function works only when data type is numerical, we will change our categorical columns to numerical, to see if there are correlation between them.

# In[13]:


changed_df = data.copy()

for col_name in changed_df.columns:
    if (changed_df[col_name].dtype == 'object'):
        changed_df[col_name] = changed_df[col_name].astype('category')
        changed_df[col_name] = changed_df[col_name].cat.codes
    
changed_df.head()


# In[14]:


data.head()


# In[15]:


corr_matrix_all = changed_df.corr() #pearson, kendall, spearman
sns.heatmap(corr_matrix_all, annot=True)

plt.title('Correlation Matrix for numerical values')

plt.show()


# In[16]:


changed_df.corr()


# In[17]:


pd.set_option('display.max_rows', None)


# It is hard to look at all those numbers, so we can use `unstack` and then filter out to look correlations which are higher then 0.5
# 

# In[18]:


cm = changed_df.corr()
corr_pairs = cm.unstack()
sorted_pairs = corr_pairs.sort_values()


# In[19]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]
high_corr


# In[20]:


data.hist(bins=50, figsize=(15,10))
plt.show()


# In[21]:


data['Rating'].unique()


# In[22]:


data.loc[(data['Rating'] == 'AO')]


# K_A(kids to adult) and E(everyone) is same thing with different names, I'm going to keep only one of them

# In[23]:


data['Rating'] = data['Rating'].replace({'K_A':'E'})


# In[24]:


data['Global_Sales'].value_counts()


# In[25]:


from pandas.plotting import scatter_matrix

attributes = ['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score',
             'User_Count']
scatter_matrix(data[attributes], figsize=(15,10))
plt.show()


# In[26]:


data['Global_Sales'].hist(bins=50)


# In[27]:


fig = plt.figure(figsize=(10,7))

ax = fig.add_axes([0,0,1,1])
ax.set_xlabel('Global Sales')

bp = ax.boxplot(data['Global_Sales'])
plt.title('Outliers in Global Sales')
plt.show()




# Our inital boxplot showed us that there are some outliers that could reduce our model accuracy, so have to  deal with them first
# Note: We can use plotly to get more information out of graphics

# Out of 7000+ entry there are only 12 games  have higher Global_sales than 20. Since number is small I'm going to delete them

# In[28]:


data.loc[(data["Global_Sales"] > 20)]


# In[29]:


data = data.drop(data[data.Global_Sales > 20].index)
data.head()


# In[30]:


import plotly.express as px
import plotly.graph_objects as go

x_val = data['Critic_Score']

y_val = data['Global_Sales']
plot = px.violin(data_frame=data, y=y_val, x=x_val)
plot.show()


# In[31]:


X = data[['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count' ]]
y = data[['Global_Sales']]


# In[41]:


#Splitting dataset with 80/20 ratio
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[33]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)


# In[42]:


#we can check our model residual plot as well. 
from yellowbrick.regressor import ResidualsPlot

res_plot = ResidualsPlot(regressor)
res_plot.fit(X_train, y_train)
res_plot.score(X_test, y_test)
res_plot.show()


# `R squared` showed in residial plot is norm of residial. Closer to the 0 better the model. 
# 
# <a target="_blank" href="https://en.wikipedia.org/wiki/Coefficient_of_determination#Comparison_with_norm_of_residuals"> CLick here for further information</a>

# In[35]:


#predicting test results
y_pred = regressor.predict(X_test)


# In[36]:


from sklearn.metrics import mean_squared_error
sale_predictions = regressor.predict(X_test)
lin_mse = mean_squared_error(y_test, sale_predictions)
lin_rmse = np.sqrt(lin_mse)
print("In our LinearRegression model's Root mean squared error is: ", lin_rmse)


# I'm going to use another model for comparison purposes

# In[37]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(X_train, y_train)


# In[38]:


sale_predictions = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, sale_predictions)
forest_rmse = np.sqrt(forest_mse)
print("In our RandomFores model's Root mean squared error is: ", forest_mse)


# Clearly Regression did better than Randomforest. Lastly we can use cross validation on RandomForest to see if our model can be better

# In[39]:


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standart deviation:', scores.std())


# In[40]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, X_test, y_test, 
                               scoring='neg_mean_squared_error',
                              cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# With Cross validation it is almost equal to LinearRegression model.

# In[ ]:




