import os
import sys
from pathlib import Path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def run():
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulation.step()


if __name__ == '__main__':
    output_path = Path('Data\Static')
    if os.path.isdir(output_path):
        pass
    else:
        os.mkdir(output_path)
    traci.start([checkBinary('sumo-gui'), '-c', 'Simulation_Environment\Static\osm.sumocfg',
    '--tripinfo-output', os.path.join(output_path,'trip-info.xml'), '--start'])
    run()