'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):
    # TODO
    entropy = 0
    
    if df.empty:
        return entropy

    target_colname = df.columns[-1] #to get last column's name
    target_values = df[target_colname].unique() # List of unique target values

    # Sumation of -pi*log(pi)
    for val in target_values:
        #value_counts() gives count of unique values
        temp1 = df[target_colname].value_counts()[val] / len(df[target_colname])
        if(temp1 == 0): #if no impurity 
            continue
        temp2 = (temp1 * np.log2(temp1))
        entropy += -temp2

    return entropy # Return Entropy(S) of dataset
    

'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    # TODO
    avg_info = 0

    if df.empty:
        return avg_info
    
    try:
        attr_values = df[attribute].unique()
        target_colname = df.columns[-1]
        target_values = df[target_colname].unique()
        
        for attr_val in attr_values:
            entropy_attr = 0
            #count of number of times variable attribute is present and store as denominator of temp
            temp_den = len(df[attribute][df[attribute] == attr_val]) 

            for tar_val in target_values:
                #no. of times attr_val and tar_val occur and store in numerator of temp
                temp_num = len(df[attribute][df[attribute] == attr_val][df[target_colname] == tar_val])
                # Find temp1
                temp1 = temp_num / temp_den
                # Accumulate Feature Entropy(A=val)
                if(temp1 == 0):
                    continue
                temp2 = (temp1 * np.log2(temp1))    
                entropy_attr += -temp2

            # Accumulate Attribute Entropy(A=val)
            avg_info += (((temp_den/len(df)) * entropy_attr))

    except KeyError:
        print(attribute," does not exist in dataset")
        
    except LookupError:
        print("Index out of bound error")
        
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):
    # TODO
    information_gain = 0

    try:
        information_gain = get_entropy_of_dataset(df) - get_avg_info_of_attribute(df,attribute)

    except KeyError:
        print(attribute," does not exist in dataset")
        
    except LookupError:
        print("Index out of bound error")

    return information_gain


#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
        '''
    # TODO
    if df.empty:
        return(dict(),'')
    
    attr_list = list(df)[:-1] #list of attributes except last one
    attr_ig = list(map(lambda x:get_information_gain(df,x),attr_list)) #to get IG of each attribute
    IG = dict(zip(attr_list,attr_ig)) #bringing to req output format
    max_IG, _ = max(enumerate(attr_ig),key = lambda x:x[1]) #_ is udes so that we can get rid of 2nd part
    col = attr_list[max_IG]
    
    return (IG,col) 
       
