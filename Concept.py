import pandas as pd
import numpy as np

def to_df(list):

def pi(df):

def time(df):

qi_df = to_df[r_id_1,r_id_2] #returns dataframe of phase's leg characteristics
#r_id|max_capacity|n

qi_df = pi(qi_df) #returns dataframe with added pi(time needed to clear leg) use eq. 1
#r_id|mx_capacity|n|pi

qi_time = time(qi_df) #returns the largest time needed to clear a leg between the two legs in the phase

if q1_time >0 or q2_time > 0:
    if q1_time > 0:
        q1 = 'G'
        q2 = 'R'
    else:
        q1 = 'R'
        q2 = 'G'

elif q1_time>0 and q2>0