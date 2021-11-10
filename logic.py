import numpy as np
import random as rd
import pandas as pd


def to_route(beginning,possible_route):
    """ 
    Returns end route depending on probability dictionary similar below
    
        possible_route = {
        
        "North":{
            "South": 40, 
            "East": 60
            },
        "South":{
            "North": 75, 
            "West": 25
            },
        "East":{
            "West": 70,
            "North": 30
            },
        "West":{
            "East": 55,
            "South": 45
            }
        }
    """
    
    direction = list(possible_route[beginning].keys())                 #creates list of keys per possible route, ex. keys of "North" is "South" and "East"
    values = list(possible_route[beginning].values())                  #creates list of values per possible route, ex. values of "North" is 40 and 60
    
    end = rd.choices(direction, weights=values, k=1)                   #to use rd.choices, lists must be assigned, not objects
    
    return end[0]

def single_load_time_index(df):
    """
    Converts single route dataframe into the time where each vehicle shall enter the lane
    """

    def when_depart(initial, final, demand):
        """
        Returns list of when each vehicle departs in seconds
        """
        #converts time boundaries to seconds to seconds
        initial_s = initial*60
        final_s = final*60

        #checks if there is demand
        if demand != 0:
            #converts from vehicle/given minutes to seconds/vehicle
            demand = (final-initial)*60/demand
            vehicles_time_index = np.arange(initial_s, final_s, step = demand, dtype='float32')
            return vehicles_time_index
        else:
            #if there is no demand, return nothing
            pass
    
    vehicle_index = np.array([])
    for row_num in range(df.shape[0]-1):
        segment_df = df.iloc[row_num:row_num+2, 0:2]    #segments the datframe by 2s in terms of row index(0-1,1-2,2-3,etc.)
        segment_df = segment_df.reset_index(drop=True)

        #use the when_depart function to analyze each segment and passes the vehicle time into the vehicle_index array 
        vehicle_time_index = when_depart(segment_df.iloc[0,0], segment_df.iloc[1,0], segment_df.iloc[1,1])
        vehicle_index = np.hstack([vehicle_index, vehicle_time_index])
    
    return vehicle_index[vehicle_index != np.array(None)]

def load_time_index(df):
    """
    Converts single route dataframe into the time where each vehicle shall enter the lane
    """
    vehicle_time_index_per_lane = {}
    

    for column_index in list(range(1,df.shape[1])):     #gets column index of routes assuming first column is time and the rest are routes
        vehicle_time_index_per_lane[df.columns.values[column_index]] = single_load_time_index(df.iloc[:,[0, column_index]]).astype('float32')
    
    return vehicle_time_index_per_lane
