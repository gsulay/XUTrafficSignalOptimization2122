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
    traci.trafficlight.setProgram('4889475255',1)
    if int(traci.trafficlight.getProgram('4889475255')) == 1:
        print('We are in the endgame bro')
    else:
        print('Buhi pa si spiderman')
        raise ValueError
    
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulation.step()
    traci.close()


if __name__ == '__main__':
    output_path = Path('Data\Actuated')
    if os.path.isdir(output_path):
        pass
    else:
        os.mkdir(output_path)

    traci.start([checkBinary('sumo-gui'), '-c', 'Simulation_Environment\Actuated\osm.sumocfg',
    '--tripinfo-output', os.path.join(output_path,'trip-info.xml'),'--start','-a','Simulation_Environment\Actuated\osm.tlLogic.xml', "--summary", r"Results\actuated.xml", "--scale", "1.0"])

    run()