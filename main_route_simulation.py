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
import logic as lc

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

    #initialize base values 
    step = 0
    current_phase_time=0
    minimum_phase_time=30
    phase_index=0
    lanearea_phasing = json_loader(Path('Data\lanearea_detector_phasing.json'))
    intersection_id = 4889475255
    max_green_time=60

    while traci.simulation.getMinExpectedNumber() > 0:
        continue
    traci.close()
    sys.stdout.flush()

    #Dars Analysis achuchu

    #Gif Edit No touchy
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