
# coding: utf-8

# # Price Precictor Model

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


housing = pd.read_csv("ML_Projects/data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[6]:



housing.hist(bins=50,figsize=(20,15))
plt.show


# ## Tain-Test Splitting

# In[7]:


# For learning purpose => This function is already written in scikit learn 
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:] 
#     return data.iloc[train_indices], data.iloc[test_indices]
# train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")




from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[8]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[9]:


strat_train_set['CHAS'].value_counts()


# In[10]:


strat_test_set['CHAS'].value_counts()


# In[11]:


# 376/28 and 95/7 gives approx 13.45 which tells that stratified suffled split splitted the datat of ['chas'] in eqaul ratio for taining and testing


# ## Looking for Correlation 

# In[12]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[13]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[14]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Trying Out Attribute Combination

# In[15]:


housing["TAXRM"] = housing['TAX']/housing['RM']
housing.head()


# In[16]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[17]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[18]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## Missing Attributes

# In[19]:


# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)


# In[20]:


a = housing.dropna(subset=["RM"]) #Option 1
a.shape
# Note that the original housing dataframe will remain unchanged


# In[21]:


housing.drop("RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[22]:


median = housing["RM"].median() # Compute median for Option 3


# In[23]:


housing["RM"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged


# In[24]:


housing.describe() # before we started filling missing attributes


# In[25]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[26]:


imputer.statistics_


# In[27]:


X = imputer.transform(housing)


# In[28]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[29]:


housing_tr.describe()


# ## Scikit-Learn Design

# Primarily, in sklearn there are three types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters
# 
# 2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.
# 
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

# ## Feature Scaling

# Primarily, two types of feature scaling methods:
# 1. Min-max scaling (Normalization)
#     (value - min)/(max - min)
#     Sklearn provides a class called MinMaxScaler for this
#     
# 2. Standardization
#     (value - mean)/std
#     Sklearn provides a class called StandardScaler for this

# ## Creating a Pipeline

# In[30]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# In[31]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[32]:


housing_num_tr.shape


# # SELECTING A DESIRED MODEL 

# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[34]:


some_data = housing.iloc[:5]


# In[35]:


some_labels = housing_labels.iloc[:5]


# In[36]:


prepared_data = my_pipeline.transform(some_data)


# In[37]:


model.predict(prepared_data)


# In[38]:


list(some_labels)


# ## Evaluating the model

# In[39]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[40]:


rmse


# ### Using better evaluation technique - Cross Validation

# In[41]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[42]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[43]:


print_scores(rmse_scores)


# ## Saving the Model 

# In[44]:


from joblib import dump, load
dump(model, 'pricePredictor.joblib') 


# # Testing the Model

# In[45]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))


# In[46]:


final_rmse


# In[47]:


prepared_data[0]


# # Using The Model 

# In[48]:


from joblib import dump, load
import numpy as np
model = load('pricePredictor.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


# In[ ]:




