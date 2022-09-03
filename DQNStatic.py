import time
import multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path
import random
from deep_q_test import SumoEnvrionment, train
import torch
import torch.nn as nn
import torch.nn.functional as F
from sumolib import checkBinary  # noqa
import traci  # noqa
import numpy as np
import pandas as pd
import os
import sys
import re
from collections import deque
from tqdm import tqdm

class Net(torch.nn.Module):
    def __init__(self, arry, states):
        super().__init__()
        self.states = states
        self.arry = arry
        self.fc1 = nn.Linear(len(self.states), 1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500, self.arry.shape[0])
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

#Only for 2 phase intersections
class SumoStaticDQN:
    def __init__(self, min_green=15, cycle=60, interval=5, gui=False, buffer=60, sumocfg_path="Simulation_Environment\Simulation Training\osm.sumocfg", scale=1.0, evaluate=False, initialize_only=False):
        #initialize required minimum values
        self.min_green      = min_green
        self.interval       = interval
        self.cycle          = cycle
        self.gui            = gui
        self.buffer         = buffer
        self.sumocfg_path   = sumocfg_path
        self.scale          = scale
        self.evaluate       = evaluate

        #Create phase array
        self.phase_arry = self.make_phase_arry(self.min_green, self.cycle, self.interval)
        
        #Initialize sumo
        self.initialize()

        #Create a log for the reward for eacg step
        self.rewards = []

        #Initialize phase program variable from phase arry
        self.current_program = 1

        #Get the TLS ID of the program
        self.tls = traci.trafficlight.getIDList()[0]
        self.initialize_phases()

        #Initialize storage values
        self.old_waiting_time = 0
        self.set_incoming_lanes()
        self.waiting_time = {}
        self.states = self.get_states()
    
    def restart(self):
        self.current_program = 1
        self.rewards = []
        self.old_waiting_time = 0
        self.waiting_time = {}
        traci.load(["-c", sumocfg_path,
                            "--tripinfo-output", "Data\\tripinfo.xml",  "--start",  "--scale", f"{scale}"])

    
    def set_incoming_lanes(self):
        self.incoming_lanes = []
        e2_detectors = traci.lanearea.getIDList()
        for detector in e2_detectors:
            incoming_lane = traci.lanearea.getLaneID(detector)
            self.incoming_lanes.append(incoming_lane)
    
    def close(self):
        traci.close()

    def initialize(self):
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        if self.gui==True:
            select_gui = checkBinary('sumo-gui')
        else:
            select_gui = checkBinary('sumo')
        
        if self.evaluate==False:
            traci.start([select_gui, "-c", self.sumocfg_path,
                                "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])
        else:
            traci.start([select_gui, "-c", self.sumocfg_path,
                                "--tripinfo-output", "Data\\tripinfo.xml",  "--start", "--summary", "Results\\dqn.xml", "--scale", f'{self.scale}'])
    
    def initialize_phases(self):
        #Find phase patterns that are not yellow phases
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls)[0]
        pattern = r'(y|Y)'
        phases_with_yellow = [i.state for i in logic.getPhases()]
        self.phases_index = [idx for idx, i in enumerate(phases_with_yellow) if re.search(pattern, i) != None]
        
    def make_phase_arry(self, min_green, cycle, interval):
        """Creates 2 phase array
        Example:\n
        array([[15, 45],\n
       [20, 40],\n
       [25, 35],\n
       [30, 30],\n
       [35, 25],\n
       [40, 20],\n
       [45, 15]])
        """
        x = np.arange(min_green, cycle-min_green+interval, interval)
        y = [cycle-i for i in x]
        arry = np.array([x,y])
        arry = np.transpose(arry)
        return arry

    def change_program(self, percentage_array):
        #Change all phases to the calculated percentage array
        for idx, i in enumerate(self.phases_index):
            traci.trafficlight.setPhase(self.tls,i)
            traci.trafficlight.setPhaseDuration(self.tls, self.cycle*percentage_array[idx])
        traci.trafficlight.setPhase(self.tls, 0)    #Set the phase of the program to the initial phase

    def action(self, array_index):
        percentage_array = self.phase_arry[array_index]
        self.change_program(percentage_array)
        self.current_program = array_index
        for i in range(self.buffer):
            traci.simulation.step()
            self.step_reward()
        
    def get_states(self):
        #gets all lane area detectors in the environment
        e2_detectors = traci.lanearea.getIDList()
        queues = [traci.lanearea.getLastStepVehicleNumber(detector) for detector in e2_detectors] #gets the queus in detectors
        #Geths 
        one_hot_vector_phase_arry = np.eye(len(self.phase_arry))[self.current_program]
        arry = np.hstack([queues, one_hot_vector_phase_arry])
        return arry
        
    def step_reward(self):
        def waiting_time():
            cars = traci.vehicle.getIDList()
            for car in cars:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car)
                road_in = traci.vehicle.getRoadID(car)
                if road_in in self.incoming_lanes:
                    self.waiting_time[car] = wait_time
                else:
                    if car in self.waiting_time:
                        del self.waiting_time[car]
            
            total_waiting_time = sum(self.waiting_time.values())
            reward = self.old_waiting_time - total_waiting_time
            self.old_waiting_time = total_waiting_time
            # print(total_waiting_time)
            return reward
        
        def queue_length():
            queue_all = np.array([])
            e2_detectors = traci.lanearea.getIDList()
            for detector in e2_detectors:
                queue = traci.lanearea.getJamLengthMeters(detector)
                queue_all = np.append(queue_all,np.float16(queue))
            return -queue_all.sum()/1000    #from m to km
        
        def check_is_max():
            max_arry = np.array([])
            e2_detectors = traci.lanearea.getIDList()
            for detector in e2_detectors:
                max_arry = np.append(max_arry, traci.lanearea.getLastStepOccupancy(detector))
            max_arry = max_arry.max()
            if max_arry > 90.0:
                return -10
            else:
                return 0
        self.rewards.append([queue_length(), check_is_max(), waiting_time()])

    def get_reward(self):
        #Converts self.rewards list to array for calculation
        arry = np.array(self.rewards)
        all = np.array([])

        #Gets the mean of all conditions
        for i in range(arry.shape[1]):
            val = arry[:,i].mean()
            all = np.append(all,val)
        
        #Resets self.rewards list
        self.rewards=[]
        return all.sum()

    def is_done(self):
        return traci.simulation.getMinExpectedNumber() == 0

def train(net=None, gui=False,scale=1.0, train=True, debug=False, epochs = 2, mem_size = 200, decay=1-1e-4,
        batch_size = 10, sync_freq = 3000, epsilon = 0.8, discount_factor=0.5, learning_rate = 1e-7, last_epoch=0, sumocfg_path="Simulation_Environment\Static DQN Simulation\osm.sumocfg"):
    starting_epsilon = epsilon
    env = SumoStaticDQN(gui=False, sumocfg_path=sumocfg_path, scale=scale, min_green=15, cycle=60, interval=5)

    #Initializes or continues the Neural Network
    if train==True:
        net = Net(env.phase_arry, env.states)
        target_net = Net(env.phase_arry, env.states)
    else:
        if net != None:
            net = net
        else:
            print('No model in variable. Please add model')
            raise AttributeError
    
    #Initialize Deep Learning
    replay              = deque(maxlen=mem_size)
    device              = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn             = nn.MSELoss()
    optimizer           = torch.optim.Adam(net.parameters(), lr=learning_rate)
    losses_progression  = []
    prev_sync           = 0

    for i in tqdm(range(epochs)):
        #Initialize the simulation
        env.restart()
        state1_ = env.get_states()
        state1 = torch.from_numpy(state1_).float().unsqueeze(dim=0).to(device)
        done=False
        net.to(device)

        while not done:
            #if in debug mode, runs simulation up to the 10,000th step
            if debug:
                if int(traci.simulation.getTime()) > 10000:
                    break
            #get the q value of the initial state
            qval = net(state1)

            #epsilon greedy policy
            if random.random() < epsilon:
                action = np.random.randint(0, len(env.phase_arry))
            else:
                action = torch.argmax(qval)
            
            env.action(action)
            state2_ = env.get_states()
            state2 = torch.from_numpy(state2_).float().unsqueeze(dim=0).to(device)

            #Save replay values
            reward = env.get_reward()
            done = env.is_done()
            exp = [state1, action, reward, state2, done]
            replay.append(exp)

            #replace state 1 with state 2 values
            state1 = state2

            if len(replay) > batch_size:
                minibatch   = random.sample(replay, batch_size)
                state1_batch    = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(device)
                action_batch    = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(device)
                reward_batch    = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(device)
                state2_batch    = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(device)
                done_batch      = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(device)

                net.to(device)
                Q1 = net(state1_batch)
                target_net.to(device)

                with torch.no_grad():
                    Q2 = target_net(state2_batch)
                
                Y = reward_batch + discount_factor*((1-done_batch)*torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()

                loss = loss_fn(X,Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses_progression.append([i,loss.item(), reward, epsilon])

                current_sync = int(traci.simulation.getTime())
                if (current_sync - prev_sync) > sync_freq:
                    epsilon = epsilon*decay
                    print(f"Syncs at {int(traci.simulation.getTime())} seconds.")
                    # print(reward)
                    target_net.load_state_dict(net.state_dict())
                    prev_sync = current_sync
            prev_sync=0
        target_net.to(torch.device('cpu'))
    local_time="_".join([str(i) for i in list(time.localtime())])

    cur_epoch = int(epochs) + int(last_epoch)
    save_path = f'DQNStatic_Model/Model-{cur_epoch}epochs-_{local_time}'
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)

    #Saves Model and Losses Data
    torch.save(target_net, os.path.join('DQNStatic_Model',Path(f'Model-{cur_epoch}epochs-_{local_time}.pth')))
    losses_progression_df = pd.DataFrame(losses_progression)
    losses_progression_df.iloc[:,0] = losses_progression_df.iloc[:,0].apply(lambda x: x + last_epoch)
    losses_progression_df.to_csv(os.path.join(save_path,Path(f'Losses.csv')))
    
    traci.close()
    args_kwargs = [f'epochs={epochs}', f'memory={mem_size}', f'e_decay={decay}', f'batch_size={batch_size}', f'sync={sync_freq}', f'e={epsilon}', f'gamma={discount_factor}', f'lr={learning_rate}']
    return target_net, losses_progression, save_path, args_kwargs

def graph_losses(losses_progress, last_epoch, save_path, args_kwargs, rolling_weight = 50):
    losses_progress = np.array(losses_progress)
    epochs = losses_progress[:,0]
    losses = losses_progress[:,1]
    rewards=losses_progress[:,2]
    epsilon = losses_progress[:,3]

    x = np.arange(0,len(losses),1)
    y = losses
    z = rewards
    e = epsilon

    df = pd.DataFrame({'Epochs':epochs, 'Episodes':x, 'Losses':y,'Rewards':z, 'Epsilon':e})
    df['Epochs'] = df['Epochs'].apply(lambda x: f"Epoch - {x + last_epoch}")
    df['Rolling Rewards'] = df['Rewards'].rolling(rolling_weight).mean()
    df['Rolling Losses'] = df['Losses'].rolling(rolling_weight).mean()
    df['Epoch and Episode'] = df.apply(lambda x: f"{x[0]}: Episode {x[1]}", axis=1)

    fig, ax = plt.subplots(nrows=3, figsize=(22,18))

    ax[0].plot(df['Epoch and Episode'], df['Losses'], label='Losses')
    ax[0].plot(df['Epoch and Episode'], df['Rolling Losses'], label=f'Rolling {rolling_weight} Episode Average')
    ax[0].set_xticks(df['Epoch and Episode'][np.arange(0,df.shape[0],int(df.shape[0]/5))])
    ax[0].legend()
    ax[0].set_title("Loss per Episode")

    ax[1].plot(df['Epoch and Episode'], df['Rewards'], label='Losses')
    ax[1].plot(df['Epoch and Episode'], df['Rolling Rewards'], label=f'Rolling {rolling_weight} Episode Average')
    ax[1].sharex(ax[0])
    ax[1].legend()
    ax[1].set_title("Rewards per Episode")

    ax[2].plot(df['Epoch and Episode'], df['Epsilon'], label='Epsilon')
    ax[2].sharex(ax[0])
    ax[2].legend()
    ax[2].set_title("Epsilon per Episode")

    ax[2].annotate(" ".join(args_kwargs), xy=(0.5,-0.2), xycoords='axes fraction', ha='center')

    plt.savefig(os.path.join(save_path, "Losses.png"))
    plt.show()

def evaluate(target_net, sumocfg_path="Simulation_Environment\Simulation Training\osm.sumocfg", scale=1.0):
    env = SumoStaticDQN(gui=True, sumocfg_path=sumocfg_path, scale=scale, min_green=15, cycle=60, interval=5)
    is_done = env.is_done()

    while not is_done:
        state_ = env.get_states()
        state = torch.from_numpy(state_).float()

        with torch.no_grad():
            qval_ = target_net(state)
        
        qval = qval_.data.numpy()
        action = np.argmax(qval)
        env.action(action)
        is_done = env.is_done()
    traci.close()

if __name__ == '__main__':
    sumocfg_path2 = "Simulation_Environment\Static DQN Simulation\osm.sumocfg"
    sumocfg_path="Simulation_Environment\Static DQN Simulation\osm.sumocfg"
    scale = 1.0
    target_net, losses, save_path, args_kwargs = train(epochs=2, gui=False, sumocfg_path=sumocfg_path2, decay=0.99, scale=scale, learning_rate=1e-5)

    t1 = multiprocessing.Process(target=graph_losses, args=[losses, 0, save_path, args_kwargs])
    t2 = multiprocessing.Process(target=evaluate, args=[target_net, sumocfg_path2], kwargs={'scale':scale})

    t1.start()
    t2.start()

    t1.join()
    t2.join()