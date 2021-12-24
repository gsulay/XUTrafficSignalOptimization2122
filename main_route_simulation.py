#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2021 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
import optparse
from libsumo.libsumo import inductionloop
from threading import Thread
from pathlib import Path

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

def json_loader(path):
    with open(path,'r') as f:
        all_data ="".join([i for i in f])
        file = json.loads(all_data)
    return file

def run():
    """executes the TraCI control loop"""
    step = 0

    #initializes the dictionary for each lane area detector to sotore data
    lanearea_det_ids = traci.lanearea.getIDList()
    lane_area_data = {}
    for sensor in lanearea_det_ids:
        lane_area_data[sensor] = np.array([])

    inductionloop_ids = traci.inductionloop.getIDList()
    inductionloop_data = {}
    for sensor in inductionloop_ids:
        inductionloop_data[sensor]=np.array([])

    #initialize base values 
    step = 0
    current_phase_time=0
    minimum_phase_time=30
    phase_index=0
    lanearea_phasing = json_loader(Path('Data\lanearea_detector_phasing.json'))
    intersection_id = 4889475255
    
    #function to read instantaneous lane area data
    def lanearea_read_data():
        for detector in lanearea_det_ids:
            measure = traci.lanearea.getLastStepVehicleNumber(detector)
            lane_area_data[detector] = np.append(lane_area_data[detector],np.array([measure],dtype='float16'))

    #function to read instantaneous induction loop data
    def inductionloop_read_data():
         for sensor in inductionloop_ids:
            measure = traci.inductionloop.getLastStepVehicleNumber(sensor)
            inductionloop_data[sensor] = np.append(inductionloop_data[sensor],np.array([measure],dtype='float16'))

    #environment for the        
    while traci.simulation.getMinExpectedNumber() > 0:
        #initializes threads list
        threads = []
        
        #creates a thread that adds the number of vehicles in each lane area detector 
        #to lanarea_det_data and induction loop data to inductionloop_data
        threads[0] = Thread(target=lanearea_read_data, args=())
        threads[0].start()
        threads[1] = Thread(target=inductionloop_read_data, args=())
        threads[1].start()

        for thread in threads:
            thread.join()
        traci.simulationStep()
        #logic for the traffic light
        #max green time = 60s, Passage time = 3s
        
        traci.trafficlight.setPhase(intersection_id,0)
        traci.simulation.step()
        
        step += 1

    
    traci.close()
    sys.stdout.flush()
    
    df = pd.DataFrame(lane_area_data)
    df['count'] = [num for num in list(range(df.shape[0]))]
    df['bin'] = pd.cut(df['count'], np.arange(0,df.shape[0],900))
    final_df = df.groupby('bin').mean()
    final_df.to_csv('Data\\data.csv')

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def get_main_directory():
    dir = os.getcwd()
    return dir


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml"])
    run()

    #puts current directory on 'Data' folder as a text file
    with open(os.path.join(os.getcwd(),'Data\\directory.txt'), 'w') as text:
        print(get_main_directory(), file=text)