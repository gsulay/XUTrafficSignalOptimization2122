import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import traci
import optparse
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from collections import deque
import random
import torch.nn.functional as F

from sumolib import checkBinary  # noqa
import traci  # noqa



class Net(nn.Module):
    def __init__(self, total_phases):
        super().__init__()
        self.total_phases = total_phases
        self.fc1 = nn.Linear(12,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,self.total_phases)

    def forward(self, x):
        #Neural Network Forward Pass Layers
        x = F.relu(self.fc1(x))
        # x = nn.BatchNorm1d(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

#Only works for micro simulation (1 traffic light)
class SumoEnvrionment:
    def __init__(self, total_phases = 4, gui = True, buffer_size = 12):
        self.total_phases = total_phases
        self.buffer_size = buffer_size
        self.gui = gui

        self.start_program()

        self.lanes_dict = {}
        self.lane_IDs = traci.lanearea.getIDList()
        for lane in self.lane_IDs:
            self.lanes_dict[lane]=[]

        self.traffic_light = traci.trafficlight.getIDList()[0]
        
    
    def sumo_initialize(self):
        options = self.get_options()
        options.nogui = self.gui
        # this script has been called from the command line. It will start sumo as a server, then connect and run
        if not options.nogui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')

        # we need to import python modules from the $SUMO_HOME/tools directory
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

    def start_program(self):
        self.sumo_initialize()
        traci.start([self.sumoBinary, "-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])

    def json_loader(self, path):
        with open(path,'r') as f:
            all_data ="".join([i for i in f])
            file = json.loads(all_data)
        return file
    
    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                            default=False, help="run the commandline version of sumo")
        
        options, args = optParser.parse_args()
        return options
    
    def update_lane_data(self):
        for lane in self.lane_IDs:
            lane_data = traci.lanearea.getLastStepHaltingNumber(lane)
            self.lanes_dict[lane].append(lane_data)
        return self.lanes_dict
    
    def save_vehicle_ids(self, lanes_dict):
        vehicle_ids=pd.DataFrame(lanes_dict)
        vehicle_ids['Steps'] = np.arange(0,len(lanes_dict[lane_IDs[0]]),1)
        vehicle_ids.to_csv('test.csv')

    def get_reward(self):
        all_speed = []
        e2_detectors = traci.lanearea.getIDList()

        for detector in e2_detectors:
            speed = traci.lanearea.getLastStepMeanSpeed(detector)
            all_speed.append(speed)
        
        return np.array(speed).mean()

    def get_state(self, trafficlight):
        e2_detectors = traci.lanearea.getIDList()

        queues = [traci.lanearea.getLastStepVehicleNumber(detector) for detector in e2_detectors] #gets the queus in detectors

        tl_phase = traci.trafficlight.getPhase(self.traffic_light)
        one_hot_vector_tl_phase = np.eye(self.total_phases)[tl_phase]
        arry = np.hstack([queues, one_hot_vector_tl_phase])

        return arry
    
    def action(self, action):
        self.lanes_dict = self.update_lane_data()
        vehicle_ids=pd.DataFrame(self.lanes_dict)
        vehicle_ids['Steps'] = np.arange(0,len(self.lanes_dict[self.lane_IDs[0]]),1)
        vehicle_ids.to_csv('test.csv')
        traci.trafficlight.setPhase(self.traffic_light, action)
        
        for i in range(self.buffer_size):
            traci.simulation.step()
    
    def is_done(self):
        return traci.simulation.getMinExpectedNumber() == 0


def train(total_phases = 4,gui=False, train=True, epochs = 80, mem_size = 1000, batch_size = 200, sync_freq = 500, epsilon = 0.2, discount_factor=0.9, learning_rate = 1e-3):
    gpu = torch.device('cuda:0')
    cpu = torch.device('cpu:0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net(total_phases)
    target_net = Net(total_phases)
    replay = deque(maxlen=mem_size)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    losses=[]
    env = SumoEnvrionment(total_phases = total_phases, gui=gui)

    for i in tqdm(range(epochs), file = sys.stdout):
        state1_ = env.get_state(env.traffic_light)
        state1 = torch.from_numpy(state1_).float().unsqueeze(dim=0)

        step = 0
        done = False

        while not done:
            net = net.to(cpu)
            step += 1
            qval = net(state1)
            qval_ = qval.data.numpy()
            if random.random() < epsilon:
                action = np.random.randint(0,4)
            else:
                action = np.argmax(qval_)

            env.action(action)
            state2_ = env.get_state(env.traffic_light)
            state2 = torch.from_numpy(state2_).float().unsqueeze(dim=0)
            reward = env.get_reward()
            done = env.is_done()
            exp = [state1, action, reward, state2, done]
            replay.append(exp)

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(gpu)
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(gpu)
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(gpu)
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(gpu)
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(gpu)

                net.to(gpu)
                Q1 = net(state1_batch)

                target_net.to(gpu)
                with torch.no_grad():
                    Q2 = target_net(state2_batch)
                
                Y = reward_batch + ((1-done_batch)*discount_factor*torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

                loss = loss_fn(X,Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                state1 = state2

                if step == sync_freq:
                    target_net.load_state_dict(net.state_dict())
            if done:
                 step = 0
        if gui == False:
            traci.close()
            traci.start([ "-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                                "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])
        if gui == True:
            traci.close()
            traci.load(["-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])

    target_net.to(torch.device('cpu'))
    torch.save(target_net, Path('DQN_Model'))
    traci.close()
    return target_net, losses

def graph_losses(losses, rolling_weight = 50):
    x = np.arange(0,len(losses),1)
    y = losses

    df = pd.DataFrame({'Epochs':x, 'Losses':y})
    df['Epochs'] = df['Epochs'].apply(lambda x: f"Epoch {x}")
    df['Rolling Losses'] = df['Losses'].rolling(rolling_weight).mean()

    fig, ax = plt.subplots()

    ax.plot(df['Epoch'], df['Losses'], label='Losses')
    ax.plot(df['Epoch'], df['Rolling Losses'], label='Rolling {rolling_weight} Epoch Average')
    
    condition = [True if i % rolling_weight == 0 else False for i in df['Epochs'].to_numpy()]
    ax.set_xticks(df['Epochs'][condition])

    ax.legend()
    plt.show()

def evaluate(target_net):
    env = SumoEnvrionment(4, gui=True)
    is_done = env.is_done()

    while not is_done:
        state_ = env.get_state()
        state = torch.from_numpy(state_).float()

        with torch.no_grad():
            qval_ = target_net(state)
        
        qval = qval_.data.to_numpy()
        action = np.argmax(qval)
        env.action(action)
        is_done = env.is_done()
        traci.close()



if __name__ == '__main__':
    target_net, losses = train(epochs=2, gui=False)
    graph_losses(losses)
    evaluate(target_net)