import time
import threading
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
    def __init__(self, states_length, total_phases):
        super().__init__()
        self.total_phases = total_phases
        self.fc1 = nn.Linear(states_length,1000)
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
    def __init__(self, gui = True, buffer_size = 15, buffer_yellow = 6):

        self.buffer_yellow = buffer_yellow
        self.buffer_size = buffer_size
        if self.buffer_size < self.buffer_yellow:
            print("Buffer size must be greater than yellow buffer")
            raise ValueError

        self.gui = gui

        self.start_program()

        self.lanes_dict = {}
        self.lane_IDs = traci.lanearea.getIDList()
        for lane in self.lane_IDs:
            self.lanes_dict[lane]=[]

        self.traffic_light = traci.trafficlight.getIDList()[0]
        self.get_phase_data(self.traffic_light)

    
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

    def get_phase_data(self, traffic_light):
        """Convert Phase SUMO XML TL Program phase data to native python string phase data
        """
        self.tl_program = traci.trafficlight.getProgram(traffic_light)
        phases_objects=traci.trafficlight.getCompleteRedYellowGreenDefinition(traffic_light)[0].getPhases()
        self.phases = [phase.state for phase in phases_objects]
        self.total_phases = len(self.phases)
        self.previous_tl_state = traci.trafficlight.getRedYellowGreenState(self.traffic_light)
        

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
        vehicle_ids['Steps'] = np.arange(0,len(lanes_dict[self.lane_IDs[0]]),1)
        vehicle_ids.to_csv('test.csv')

    def get_reward(self):
        all_speed = []
        e2_detectors = traci.lanearea.getIDList()

        for detector in e2_detectors:
            speed = traci.lanearea.getLastStepMeanSpeed(detector)
            if speed < 0:
                speed = 0
            all_speed.append(speed)

        population = len(all_speed)
        all_speed = np.array(all_speed)
        speed_max = all_speed.max()
        if speed_max == 0:
            return 0
        speed_max_ratio = all_speed/speed_max
        speed_max_sum = speed_max_ratio.sum()

        return (1/population)*speed_max_sum

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
        
        #creates a function that turns green going red to yellow states
        def change_green_to_yellow(to,prev):
            green = (prev=='G' or prev=='g')
            if to=='r' and green:
                return 'y'
            else:
                return prev

        true_state_len = self.buffer_size - self.buffer_yellow

        to_phase = self.phases[action]    # gets current tl state
        compare = zip(to_phase,self.previous_tl_state)  #zips the current abd previous states for comparison
        buffer_state = "".join([change_green_to_yellow(t,p) for t,p in compare])
        
        traci.trafficlight.setRedYellowGreenState(self.traffic_light, buffer_state)
        for i in range(self.buffer_yellow):
            traci.simulation.step()

        traci.trafficlight.setRedYellowGreenState(self.traffic_light, self.phases[action])
        for i in range(true_state_len):
            traci.trafficlight.setRedYellowGreenState(self.traffic_light, self.phases[action])
            
            traci.simulation.step()
        self.previous_tl_state = to_phase #sets the already finished state as the new prev state
    
    
    def is_done(self):
        return traci.simulation.getMinExpectedNumber() == 0


def train(gui=False, train=True, debug=False, epochs = 80, mem_size = 1500, batch_size = 800, sync_freq = 500, epsilon = 0.3, discount_factor=0.9, learning_rate = 1e-4):

    env = SumoEnvrionment(gui=gui)
    states_length = len(env.get_state(env.traffic_light))

    net = Net(states_length,env.total_phases)
    target_net = Net(states_length,env.total_phases)

    replay = deque(maxlen=mem_size)
    
    
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    losses_progression=[]
    prev_sync = 0
    for i in tqdm(range(epochs), file = sys.stdout):
        state1_ = env.get_state(env.traffic_light)
        state1 = torch.from_numpy(state1_).float().unsqueeze(dim=0).to(device)

        step = 0
        done = False
        net.to(device)
        while not done:

            #For debugging purposes only
            if debug:
                if int(traci.simulation.getTime()) > 10000:
                    break

            qval = net(state1)
            # qval_ = qval.data.numpy()
            if random.random() < epsilon:
                action = np.random.randint(0,env.total_phases)
            else:
                action = torch.argmax(qval)
            
            env.action(action)
            state2_ = env.get_state(env.traffic_light)
            state2 = torch.from_numpy(state2_).float().unsqueeze(dim=0).to(device)
            reward = env.get_reward()
            done = env.is_done()
            exp = [state1, action, reward, state2, done]
            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)

                net.to(device)
                Q1 = net(state1_batch)

                target_net.to(device)
                with torch.no_grad():
                    Q2 = target_net(state2_batch)
                
                Y = reward_batch + ((1-done_batch)*discount_factor*torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

                # if there are is colliding vehicles, set reward to 0
                if traci.simulation.getCollidingVehiclesNumber() > 0:
                    print(f"{traci.simulation.getCollidingVehiclesNumber()} crashed.")
                    Y = torch.zeros(Y.shape)

                loss = loss_fn(X,Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses_progression.append([i,loss.item()])

                current_sync = int(traci.simulation.getTime())
                if (current_sync - prev_sync) > sync_freq:
                    print(f"Syncs at {int(traci.simulation.getTime())} seconds.")
                    target_net.load_state_dict(net.state_dict())
                    prev_sync = current_sync
            if done:
                 step = 0
                 prev_sync = 0
        if gui == False:
            traci.close()
            traci.start([env.sumoBinary, "-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])
        if gui == True:
            traci.close()
            traci.load(["-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])

    target_net.to(torch.device('cpu'))
    local_time="_".join([str(i) for i in list(time.localtime())])
    torch.save(target_net, Path(f'DQN_Model/Model-{epochs}epochs-_{local_time}.pth'))
    traci.close()
    return target_net, losses_progression

def continue_train(net, last_epoch, gui=False, train=True, debug=False, epochs = 80, mem_size = 1500, batch_size = 800, sync_freq = 500, epsilon = 0.3, discount_factor=0.9, learning_rate = 1e-4):
    env = SumoEnvrionment(gui = gui)
    target_net.load_state_dict(net.state_dict())

    replay = deque(maxlen=mem_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    losses_progression=[]
    prev_sync = 0
    for i in tqdm(range(epochs), file = sys.stdout):
        state1_ = env.get_state(env.traffic_light)
        state1 = torch.from_numpy(state1_).float().unsqueeze(dim=0).to(device)

        step = 0
        done = False
        net.to(device)
        while not done:

            #For debugging purposes only
            if debug:
                if int(traci.simulation.getTime()) > 10000:
                    break

            qval = net(state1)
            # qval_ = qval.data.numpy()
            if random.random() < epsilon:
                action = np.random.randint(0,env.total_phases)
            else:
                action = torch.argmax(qval)
            
            env.action(action)
            state2_ = env.get_state(env.traffic_light)
            state2 = torch.from_numpy(state2_).float().unsqueeze(dim=0).to(device)
            reward = env.get_reward()
            done = env.is_done()
            exp = [state1, action, reward, state2, done]
            replay.append(exp)
            state1 = state2

            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)

                net.to(device)
                Q1 = net(state1_batch)

                target_net.to(device)
                with torch.no_grad():
                    Q2 = target_net(state2_batch)
                
                Y = reward_batch + ((1-done_batch)*discount_factor*torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

                # if there are is colliding vehicles, set reward to 0
                if traci.simulation.getCollidingVehiclesNumber() > 0:
                    print(f"{traci.simulation.getCollidingVehiclesNumber()} crashed.")
                    Y = torch.zeros(Y.shape)

                loss = loss_fn(X,Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses_progression.append([i,loss.item()])

                current_sync = int(traci.simulation.getTime())
                if (current_sync - prev_sync) > sync_freq:
                    print(f"Syncs at {int(traci.simulation.getTime())} seconds.")
                    target_net.load_state_dict(net.state_dict())
                    prev_sync = current_sync
            if done:
                 step = 0
                 prev_sync = 0
        if gui == False:
            traci.close()
            traci.start([env.sumoBinary, "-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])
        if gui == True:
            traci.close()
            traci.load(["-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])

    target_net.to(torch.device('cpu'))
    local_time="_".join([str(i) for i in list(time.localtime())])
    torch.save(target_net, Path(f'DQN_Model/Model-{last_epoch + epochs}epochs-_{local_time}.pth'))
    traci.close()
    return target_net, losses_progression


def graph_losses(losses_progress, last_epoch, rolling_weight = 50):
    losses_progress = np.array(losses_progress)
    epochs = losses_progress[:,0]
    losses = losses_progress[:,1]

    x = np.arange(0,len(losses),1)
    y = losses

    df = pd.DataFrame({'Epochs':epochs, 'Episodes':x, 'Losses':y})
    df['Epochs'] = df['Epochs'].apply(lambda x: f"Epoch - {x + last_epoch}")
    df['Rolling Losses'] = df['Losses'].rolling(rolling_weight).mean()
    df['Epoch and Episode'] = df.apply(lambda x: f"{x[0]}: Episode {x[1]}", axis=1)

    fig, ax = plt.subplots(figsize=(22,10))

    ax.plot(df['Epoch and Episode'], df['Losses'], label='Losses')
    ax.plot(df['Epoch and Episode'], df['Rolling Losses'], label=f'Rolling {rolling_weight} Episode Average')
    
    ax.set_xticks(df['Epoch and Episode'][np.arange(0,df.shape[0],int(df.shape[0]/5))])

    ax.legend()
    plt.show()
    local_time="_".join([str(i) for i in list(time.localtime())])
    plt.savefig(Path(f"Figures\Losses\Model-{epochs[-1]}_epochs-_{local_time}"))

def evaluate(target_net):
    env = SumoEnvrionment(gui=True)
    is_done = env.is_done()

    while not is_done:
        state_ = env.get_state(env.traffic_light)
        state = torch.from_numpy(state_).float()

        with torch.no_grad():
            qval_ = target_net(state)
        
        qval = qval_.data.numpy()
        action = np.argmax(qval)
        env.action(action)
        is_done = env.is_done()
    traci.close()



if __name__ == '__main__':
    target_net, losses = train(epochs=2, gui=False)

    # t1 = threading.Thread(target=graph_losses, args=[losses, 0])
    t2 = threading.Thread(target=evaluate, args=[target_net])

    # t1.start()
    t2.start()

    # t1.join()
    t2.join()