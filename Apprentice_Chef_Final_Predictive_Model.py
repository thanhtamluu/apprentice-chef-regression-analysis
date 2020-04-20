#!/usr/bin/env python
# coding: utf-8

# In[1]:


# timeit

# Student Name : Thanh Tam Luu
# Cohort       : FMSBA2


################################################################################
# Import Packages
################################################################################

# importing libraries
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # essential graphical output
import seaborn as sns # enhanced graphical output
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split # train/test split
from sklearn.ensemble import GradientBoostingRegressor # gradient boosting regressor


# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


################################################################################
# Load Data
################################################################################

# specifying file name
file = 'Apprentice_Chef_Dataset.xlsx'


# reading the file into Python
original_df = pd.read_excel(file)


################################################################################
# Feature Engineering
################################################################################

# Working with additional information from the case study

# Creating a column for the price per each meal

original_df['PRICE_PER_MEAL'] = original_df['REVENUE'] / original_df['TOTAL_MEALS_ORDERED']


# Outlier Analysis

# Setting outlier thresholds
REVENUE_hi                     = 5100
TOTAL_MEALS_ORDERED_hi         = 300 
UNIQUE_MEALS_PURCH_hi          = 9    
CONTACTS_W_CUSTOMER_SERVICE_hi = 12           
AVG_TIME_PER_SITE_VISIT_hi     = 230                    
CANCELLATIONS_BEFORE_NOON_hi   = 6       
CANCELLATIONS_AFTER_NOON_hi    = 2              
MOBILE_LOGINS_lo               = 5  
MOBILE_LOGINS_hi               = 6  
PC_LOGINS_lo                   = 1
PC_LOGINS_hi                   = 2                     
EARLY_DELIVERIES_hi            = 9    
LATE_DELIVERIES_hi             = 8.5     
AVG_PREP_VID_TIME_hi           = 280     
LARGEST_ORDER_SIZE_lo          = 2
LARGEST_ORDER_SIZE_hi          = 7
MASTER_CLASSES_ATTENDED_hi     = 2   
MEDIAN_MEAL_RATING_lo          = 2   
MEDIAN_MEAL_RATING_hi          = 4  
AVG_CLICKS_PER_VISIT_hi        = 17.5  
TOTAL_PHOTOS_VIEWED_hi         = 375
PRICE_PER_MEAL_hi              = 23      # values above this point indicates an extra beverage purchase


# Developing features (columns) for outliers

# TOTAL_MEALS_ORDERED
original_df['out_TOTAL_MEALS_ORDERED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_hi]

original_df['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

# UNIQUE_MEALS_PURCH
original_df['out_UNIQUE_MEALS_PURCH'] = 0
condition_hi = original_df.loc[0:,'out_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_hi]

original_df['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
original_df['out_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = original_df.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_hi]

original_df['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

# AVG_TIME_PER_SITE_VISIT
original_df['out_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_hi]

original_df['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)

# CANCELLATIONS_BEFORE_NOON
original_df['out_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_hi]

original_df['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                                     value      = 1,
                                                     inplace    = True)

# CANCELLATIONS_AFTER_NOON
original_df['out_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = original_df.loc[0:,'out_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_hi]

original_df['out_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                                    value      = 1,
                                                    inplace    = True)

# MOBILE_LOGINS
original_df['out_MOBILE_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] > MOBILE_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_MOBILE_LOGINS'][original_df['MOBILE_LOGINS'] < MOBILE_LOGINS_lo]

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                         value      = 1,
                                         inplace    = True)

original_df['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                         value      = 1,
                                         inplace    = True)

# PC_LOGINS
original_df['out_PC_LOGINS'] = 0
condition_hi = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] > PC_LOGINS_hi]
condition_lo = original_df.loc[0:,'out_PC_LOGINS'][original_df['PC_LOGINS'] < PC_LOGINS_lo]

original_df['out_PC_LOGINS'].replace(to_replace = condition_hi,
                                     value      = 1,
                                     inplace    = True)

original_df['out_PC_LOGINS'].replace(to_replace = condition_lo,
                                     value      = 1,
                                     inplace    = True)

# EARLY_DELIVERIES 
original_df['out_EARLY_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_EARLY_DELIVERIES'][original_df['EARLY_DELIVERIES'] > EARLY_DELIVERIES_hi]

original_df['out_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                        value      = 1,
                                        inplace    = True)

# LATE_DELIVERIES
original_df['out_LATE_DELIVERIES'] = 0
condition_hi = original_df.loc[0:,'out_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_hi]

original_df['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                       value      = 1,
                                       inplace    = True)

# AVG_PREP_VID_TIME
original_df['out_AVG_PREP_VID_TIME'] = 0
condition_hi = original_df.loc[0:,'out_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_hi]

original_df['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                         value      = 1,
                                         inplace    = True)

# LARGEST_ORDER_SIZE
original_df['out_LARGEST_ORDER_SIZE'] = 0
condition_hi = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_hi]
condition_lo = original_df.loc[0:,'out_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] < LARGEST_ORDER_SIZE_lo]

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)

original_df['out_LARGEST_ORDER_SIZE'].replace(to_replace = condition_lo,
                                              value      = 1,
                                              inplace    = True)

# MASTER_CLASSES_ATTENDED
original_df['out_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = original_df.loc[0:,'out_MASTER_CLASSES_ATTENDED'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_hi]

original_df['out_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)

# MEDIAN_MEAL_RATING
original_df['out_MEDIAN_MEAL_RATING'] = 0
condition_hi = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_hi]
condition_lo = original_df.loc[0:,'out_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] < MEDIAN_MEAL_RATING_lo]

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)

original_df['out_MEDIAN_MEAL_RATING'].replace(to_replace = condition_lo,
                                              value      = 1,
                                              inplace    = True)

# AVG_CLICKS_PER_VISIT
original_df['out_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = original_df.loc[0:,'out_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_hi]

original_df['out_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                                value      = 1,
                                                inplace    = True)

# TOTAL_PHOTOS_VIEWED
original_df['out_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = original_df.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_hi]

original_df['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

# PRICE_PER_MEAL
original_df['out_PRICE_PER_MEAL'] = 0
condition_hi = original_df.loc[0:,'out_PRICE_PER_MEAL'][original_df['PRICE_PER_MEAL'] > PRICE_PER_MEAL_hi]

original_df['out_PRICE_PER_MEAL'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)


# Setting trend-based thresholds
TOTAL_MEALS_ORDERED_change_hi          = 200
UNIQUE_MEALS_PURCH_change_hi           = 9
CONTACTS_W_CUSTOMER_SERVICE_change_hi  = 10
AVG_TIME_PER_SITE_VISIT_change_hi      = 210
CANCELLATIONS_BEFORE_NOON_change_hi    = 7
CANCELLATIONS_AFTER_NOON_change_hi     = 2
LATE_DELIVERIES_change_hi              = 10
AVG_PREP_VID_TIME_change_hi            = 280
LARGEST_ORDER_SIZE_change_hi           = 6
MASTER_CLASSES_ATTENDED_change_hi      = 1
MEDIAN_MEAL_RATING_change_hi           = 4
AVG_CLICKS_PER_VISIT_change_lo         = 10
AVG_CLICKS_PER_VISIT_change_hi         = 18
TOTAL_PHOTOS_VIEWED_change_hi          = 350

# Trend-based features

# TOTAL_MEALS_ORDERED
original_df['change_TOTAL_MEALS_ORDERED'] = 0
condition_change_hi = original_df.loc[0:,'change_TOTAL_MEALS_ORDERED'][original_df['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_change_hi]

original_df['change_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_change_hi,
                                                  value      = 1,
                                                  inplace    = True)


# UNIQUE_MEALS_PURCH
original_df['change_UNIQUE_MEALS_PURCH'] = 0
condition_change_hi = original_df.loc[0:,'change_UNIQUE_MEALS_PURCH'][original_df['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_change_hi]
original_df['change_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_change_hi,
                                                 value      = 1,
                                                 inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
original_df['change_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_change_hi = original_df.loc[0:,'change_CONTACTS_W_CUSTOMER_SERVICE'][original_df['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_change_hi]
original_df['change_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_change_hi,
                                                          value      = 1,
                                                          inplace    = True)


# AVG_TIME_PER_SITE_VISIT
original_df['change_AVG_TIME_PER_SITE_VISIT'] = 0
condition_change_hi = original_df.loc[0:,'change_AVG_TIME_PER_SITE_VISIT'][original_df['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_change_hi]

original_df['change_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_change_hi,
                                                      value      = 1,
                                                      inplace    = True)


# CANCELLATIONS_BEFORE_NOON
original_df['change_CANCELLATIONS_BEFORE_NOON'] = 0
condition_change_hi = original_df.loc[0:,'change_CANCELLATIONS_BEFORE_NOON'][original_df['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_change_hi]


original_df['change_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_change_hi,
                                                        value      = 1,
                                                        inplace    = True)


# CANCELLATIONS_AFTER_NOON
original_df['change_CANCELLATIONS_AFTER_NOON'] = 0
condition_change_hi = original_df.loc[0:,'change_CANCELLATIONS_AFTER_NOON'][original_df['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_change_hi]
original_df['change_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_change_hi,
                                                       value      = 1,
                                                       inplace    = True)


# LATE_DELIVERIES
original_df['change_LATE_DELIVERIES'] = 0
condition_change_hi = original_df.loc[0:,'change_LATE_DELIVERIES'][original_df['LATE_DELIVERIES'] > LATE_DELIVERIES_change_hi]

original_df['change_LATE_DELIVERIES'].replace(to_replace = condition_change_hi,
                                              value      = 1,
                                              inplace    = True)


# AVG_PREP_VID_TIME
original_df['change_AVG_PREP_VID_TIME'] = 0
condition_change_hi = original_df.loc[0:,'change_AVG_PREP_VID_TIME'][original_df['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_change_hi]

original_df['change_AVG_PREP_VID_TIME'].replace(to_replace = condition_change_hi,
                                                value      = 1,
                                                inplace    = True)


# LARGEST_ORDER_SIZE
original_df['change_LARGEST_ORDER_SIZE'] = 0
condition_change_hi = original_df.loc[0:,'change_LARGEST_ORDER_SIZE'][original_df['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_change_hi]

original_df['change_LARGEST_ORDER_SIZE'].replace(to_replace = condition_change_hi,
                                                 value      = 1,
                                                 inplace    = True)

# MASTER_CLASSES_ATTENDED
original_df['change_MASTER_CLASSES_ATTENDED'] = 0
condition_change_hi = original_df.loc[0:,'change_MASTER_CLASSES_ATTENDED'][original_df['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_change_hi]

original_df['change_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_change_hi,
                                                 value      = 1,
                                                 inplace    = True)

# MEDIAN_MEAL_RATING
original_df['change_MEDIAN_MEAL_RATING'] = 0
condition_change_hi = original_df.loc[0:,'change_MEDIAN_MEAL_RATING'][original_df['MEDIAN_MEAL_RATING'] > MEDIAN_MEAL_RATING_change_hi]

original_df['change_MEDIAN_MEAL_RATING'].replace(to_replace = condition_change_hi,
                                                 value      = 1,
                                                 inplace    = True)


# AVG_CLICKS_PER_VISIT
original_df['change_AVG_CLICKS_PER_VISIT'] = 0
condition_change_lo = original_df.loc[0:,'change_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_change_hi]
condition_change_hi = original_df.loc[0:,'change_AVG_CLICKS_PER_VISIT'][original_df['AVG_CLICKS_PER_VISIT'] < AVG_CLICKS_PER_VISIT_change_lo]

original_df['change_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_change_hi,
                                                   value      = 1,
                                                   inplace    = True)
original_df['change_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_change_lo,
                                                   value      = 1,
                                                   inplace    = True)

# TOTAL_PHOTOS_VIEWED
original_df['change_TOTAL_PHOTOS_VIEWED'] = 0
condition_change_hi = original_df.loc[0:,'change_TOTAL_PHOTOS_VIEWED'][original_df['TOTAL_PHOTOS_VIEWED'] > TOTAL_PHOTOS_VIEWED_change_hi]

original_df['change_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_change_hi,
                                                  value      = 1,
                                                  inplace    = True)


# Working with Email Addresses

# Step 1: Splitting personal emails 

# Placeholder list
placeholder_lst = []  

# Looping over each email address
for index, col in original_df.iterrows(): 
    
    # Splitting email domain at '@'
    split_email = original_df.loc[index, 'EMAIL'].split(sep = '@') 
    
    # Appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# Converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)


# Displaying the results
email_df


# Step 2: Concatenating with original DataFrame

# Renaming column to concatenate
email_df.columns = ['NAME' , 'EMAIL_DOMAIN']


# Concatenating personal_email_domain with friends DataFrame 
original_df = pd.concat([original_df, email_df.loc[:, 'EMAIL_DOMAIN']], 
                   axis = 1)


# Printing value counts of personal_email_domain
original_df.loc[: ,'EMAIL_DOMAIN'].value_counts()


# Step 3: One hot encoding categorical variables
one_hot_EMAIL_DOMAIN = pd.get_dummies(original_df['EMAIL_DOMAIN'])

# Dropping categorical variables after they've been encoded
original_df = original_df.drop('EMAIL_DOMAIN', axis = 1)


# Joining codings together
original_df = original_df.join([one_hot_EMAIL_DOMAIN])


# Saving new columns
new_columns = original_df.columns


################################################################################
# Train/Test Split
################################################################################

# Preparing explanatory variable data
original_df_data   = original_df.drop(['REVENUE'],
                                       axis = 1)


# Preparing response variable data
original_df_target = original_df.loc[:, 'REVENUE']


# Preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            original_df_data,
            original_df_target,
            test_size = 0.25,
            random_state = 222)


# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


# Declaring set of x-variables
x_variables = ['TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE',
               'AVG_PREP_VID_TIME', 'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED', 
               'MEDIAN_MEAL_RATING', 'TOTAL_PHOTOS_VIEWED', 'out_UNIQUE_MEALS_PURCH', 'out_MASTER_CLASSES_ATTENDED',
               'out_MEDIAN_MEAL_RATING', 'change_UNIQUE_MEALS_PURCH', 'change_CONTACTS_W_CUSTOMER_SERVICE', 
               'change_MEDIAN_MEAL_RATING', 'unitedhealth.com', 'PRICE_PER_MEAL', 'out_PRICE_PER_MEAL']


# Looping to make x-variables suitable for statsmodels
for val in x_variables:
    print(f"original_df_train['{val}'] +")


# Merging X_train and y_train so that they can be used in statsmodels
original_df_train = pd.concat([X_train, y_train], axis = 1)


# Step 1: Building a model
lm_best = smf.ols(formula =  """REVENUE ~ original_df_train['TOTAL_MEALS_ORDERED'] +
                                          original_df_train['UNIQUE_MEALS_PURCH'] +
                                          original_df_train['CONTACTS_W_CUSTOMER_SERVICE'] +
                                          original_df_train['AVG_PREP_VID_TIME'] +
                                          original_df_train['LARGEST_ORDER_SIZE'] +
                                          original_df_train['MASTER_CLASSES_ATTENDED'] +
                                          original_df_train['MEDIAN_MEAL_RATING'] +
                                          original_df_train['TOTAL_PHOTOS_VIEWED'] +
                                          original_df_train['out_UNIQUE_MEALS_PURCH'] +
                                          original_df_train['out_MASTER_CLASSES_ATTENDED'] +
                                          original_df_train['out_MEDIAN_MEAL_RATING'] +
                                          original_df_train['change_UNIQUE_MEALS_PURCH'] +
                                          original_df_train['change_CONTACTS_W_CUSTOMER_SERVICE'] +
                                          original_df_train['change_MEDIAN_MEAL_RATING'] +
                                          original_df_train['unitedhealth.com'] +
                                          original_df_train['PRICE_PER_MEAL'] +
                                          original_df_train['out_PRICE_PER_MEAL']""",
                                data = original_df_train)


# Step 2: Fitting the model based on the data
results = lm_best.fit()



# Step 3: Analyze the summary output
print(results.summary())


# Applying model in scikit-learn

# Preparing a DataFrame based the the analysis above
original_df_data   = original_df.loc[ : , x_variables]


# Preparing the target variable
original_df_target = original_df.loc[:, 'REVENUE']


# Running train/test split again
X_train, X_test, y_train, y_test = train_test_split(
            original_df_data,
            original_df_target,
            test_size = 0.25,
            random_state = 222)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING a model object
gbt = GradientBoostingRegressor()

# FITTING to the training data
gbt.fit(X_train,y_train)

# PREDICTING on new data
gbt.predict(X_test)


################################################################################
# Final Model Score (score)
################################################################################

# SCORING the results
print('Training Score:', gbt.score(X_train, y_train).round(4))
print('Testing Score:',  gbt.score(X_test, y_test).round(4))

# saving scoring data for future use
train_score = gbt.score(X_train, y_train).round(4)
test_score  = gbt.score(X_test, y_test).round(4)


# In[ ]:




