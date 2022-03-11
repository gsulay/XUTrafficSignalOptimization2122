import traci
from sumolib import checkBinary  # noqa
import optparse
import os
import sys

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

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

with open("test.txt", "w") as f:

    f.write(str(traci.trafficlight.getCompleteRedYellowGreenDefinition(tlsID=traci.trafficlight.getIDList()[0])))