import pandas as pd
import numpy as np
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#Create DF function that uses oneHotEncoder to encode the categorical columns

def buildDataFrame():
    
    churn = pd.read_csv('BankChurners.csv')
    churn['Attrition_Flag'] = churn['Attrition_Flag'].apply(encodeAttrition)
    churn['Gender'] = churn['Gender'].apply(encodeGender)
    
    
    #OneHotEncoder
    enc = OneHotEncoder(sparse=False)
    #Creating a list of the columns with categorical variables

    categorical_variables = ['Education_Level','Marital_Status','Income_Category','Card_Category']
    encoded_data = enc.fit_transform(churn[categorical_variables])
    encoded_df = pd.DataFrame(
    encoded_data,
    columns = enc.get_feature_names(categorical_variables)
    )
    #Scaling the numerical columns
    churn_scaled = StandardScaler().fit_transform(churn[['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                         'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1']])
    #Create a dataframe with the scaled data
    churn_transformed = pd.DataFrame(churn_scaled, columns=['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                            'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1'])
    
    
    churn_concat = pd.concat(
        [churn[['Attrition_Flag','Gender']],churn_transformed,
         encoded_df,
        ],
        axis=1
)
    return churn_concat
     

def encodeAttrition(Attrition_Type):
    """
    This will encode the column so 1 is Existing Customer and 0 is Attrited Customer
    """
    if Attrition_Type == 'Existing Customer':
        return 0
    else: 
        return 1
    

#Encoding gender column
def encodeGender(gender):
    """
    encoding so 1 = M and 0 = F
    """
    if gender == 'M':
        return 1
    else:
        return 0

#Create DF function that uses get_dummies to encode the categorical columns

def buildDataFrame_2():
    
    
    churn = pd.read_csv('BankChurners.csv')
    churn['Attrition_Flag'] = churn['Attrition_Flag'].apply(encodeAttrition)
    churn['Gender'] = churn['Gender'].apply(encodeGender)
    
    card_dummies = pd.get_dummies(churn[['Education_Level','Marital_Status','Income_Category','Card_Category']])
    dummies_df = pd.DataFrame(card_dummies)
    
    #StandardScaler 
    churn_scaled = StandardScaler().fit_transform(churn[['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                         'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1']])
    
    #Create a dataframe with the scaled data
    churn_transformed = pd.DataFrame(churn_scaled, columns=['Total_Relationship_Count','Contacts_Count_12_mon','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1',
                                                            'Total_Trans_Ct','Total_Ct_Chng_Q4_Q1'])
    
    
    
    
    churn_dummies_concat = pd.concat(
        [churn[['Attrition_Flag','Gender']],churn_transformed,
         dummies_df
                 ],
        axis=1
)
    return churn_dummies_concat

def encodeAttrition(Attrition_Type):
    """
    This will encode the column so 1 is Existing Customer and 0 is Attrited Customer
    """
    if Attrition_Type == 'Existing Customer':
        return 0
    else: 
        return 1
    

#Encoding gender column
def encodeGender(gender):
    """
    encoding so 1 = M and 0 = F
    """
    if gender == 'M':
        return 1
    else:
        return 0
