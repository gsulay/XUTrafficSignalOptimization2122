import time
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import optparse
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from collections import deque
import random
import torch.nn.functional as F
import multiprocessing

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
        x = self.fc3(x)
        return x

#Only works for micro simulation (1 traffic light)
class SumoEnvrionment:
    def __init__(self, gui = True, buffer_size = 15, buffer_yellow = 5, max_green=60, sumocfg_path="Simulation_Environment\Simulation Training\osm.sumocfg", evaluate=False, scale=1.0):
        self.scale = scale
        self.evaluate = evaluate
        self.sumocfg_path = sumocfg_path
        self.buffer_yellow = buffer_yellow
        self.buffer_size = buffer_size
        self.max_green = max_green
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
        self.teleported_number = traci.simulation.getStartingTeleportNumber()
        
        self.waiting_time = {}
        self.set_incoming_lanes()
        self.sum_waiting_time = 0
        self.old_waiting_time = 0
        self.counter = 0
    
    def reset(self):
        self.all_speed = []
        self.teleported_number = traci.simulation.getStartingTeleportNumber()
        self.waiting_time = {}
        self.set_incoming_lanes()
        self.sum_waiting_time = 0
        self.old_waiting_time = 0
        self.counter = 0
        self.prev_phase = 0


    def set_incoming_lanes(self):
        self.incoming_lanes = []
        e2_detectors = traci.lanearea.getIDList()
        for detector in e2_detectors:
            incoming_lane = traci.lanearea.getLaneID(detector)
            self.incoming_lanes.append(incoming_lane)
    
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
        print(self.phases)
        

    def start_program(self):
        self.sumo_initialize()
        if not self.evaluate:
            traci.start([self.sumoBinary, "-c", self.sumocfg_path,
                                "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])
        else:
            traci.start([self.sumoBinary, "-c", self.sumocfg_path,
                                "--tripinfo-output", "Data\\tripinfo.xml",  "--start", "--summary", "Results\\dqn.xml", "--scale", f'{self.scale}'])
    
    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                            default=False, help="run the commandline version of sumo")
        
        options, args = optParser.parse_args()
        return options
    
    # def update_lane_data(self):
    #     for lane in self.lane_IDs:
    #         lane_data = traci.lanearea.getLastStepHaltingNumber(lane)
    #         self.lanes_dict[lane].append(lane_data)
    #     return self.lanes_dict
    
    def save_vehicle_ids(self, lanes_dict):
        vehicle_ids=pd.DataFrame(lanes_dict)
        vehicle_ids['Steps'] = np.arange(0,len(lanes_dict[self.lane_IDs[0]]),1)
        vehicle_ids.to_csv('test.csv')

    def get_reward(self):
        def positive_speed():
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

        def pos_alter():
            all_speed = []
            vehicles = traci.vehicle.getIDList()
            for vehicle in vehicles:
                speed = traci.vehicle.getSpeed(vehicle)
                if speed < 0:
                    speed = 0
                road_in = traci.vehicle.getRoadID(vehicle)
                if road_in in self.incoming_lanes:
                    all_speed.append(speed)
            try:
                population = len(all_speed)
                all_speed = np.array(all_speed)
                speed_max = all_speed.max()
                if speed_max == 0:
                    return 0
                speed_max_ratio = all_speed/speed_max
                speed_max_sum = speed_max_ratio.sum()

                return (1/population)*speed_max_sum
            except ValueError:
                return 0

            
        def true_average_speed():
            all_speed = []
            e2_detectors = traci.lanearea.getIDList()

            for detector in e2_detectors:
                vehicles = traci.lanearea.getLastStepVehicleIDs(detector)
                for vehicle in vehicles:
                    speed = traci.vehicle.getSpeed(vehicle)
                    if speed < 0:
                        speed = 0
                    all_speed.append(speed)
            all_speed = np.array(all_speed)

            reward = all_speed.sum()/len(all_speed) #get speed from m/s to km/h
    
            return reward
        
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
        
        def over_max_green():
            if self.counter > self.max_green:
                # print('Over Max Green')
                return -10
            else:
                return 0

        #Edit result to use different reward functios
        #Pure negative
        result = queue_length() + check_is_max() + waiting_time()
        
        # With positive
        # if check_is_max() < 0:
        #     print('Punishing System')
        #     result=-10
        # else:
        #     result = positive_speed() + waiting_time()

        #however, if reward causes extreme queue length, severely punish the system

        # print(result)
        return result

    def get_state(self, trafficlight):
        e2_detectors = traci.lanearea.getIDList()
        queues = [traci.lanearea.getLastStepVehicleNumber(detector) for detector in e2_detectors] #gets the queus in detectors

        tl_phase = self.phases.index(traci.trafficlight.getRedYellowGreenState(self.traffic_light))
        one_hot_vector_tl_phase = np.eye(self.total_phases)[tl_phase]
        arry = np.hstack([queues, one_hot_vector_tl_phase])

        return arry
    
    def action(self, action):
        
        #creates a function that turns green going red to yellow states
        def change_green_to_yellow(to,prev):
            green = (prev=='G' or prev=='g')
            if to=='r' and green:
                return 'y'
            else:
                return prev

        true_state_len = self.buffer_size - self.buffer_yellow

        to_phase = self.phases[action]    # gets current tl state

        if self.previous_tl_state == to_phase:
            self.counter += self.buffer_size

        else:
            self.counter = 0


        compare = zip(to_phase,self.previous_tl_state)  #zips the current abd previous states for comparison
        buffer_state = "".join([change_green_to_yellow(t,p) for t,p in compare])
        
        traci.trafficlight.setRedYellowGreenState(self.traffic_light, buffer_state)
        for i in range(self.buffer_yellow):
            traci.simulation.step()

        traci.trafficlight.setRedYellowGreenState(self.traffic_light, self.phases[action])
        for i in range(true_state_len):
            traci.simulation.step()
        self.previous_tl_state = to_phase #sets the already finished state as the new prev state
    
    
    def is_done(self):
        return traci.simulation.getMinExpectedNumber() == 0


def train(net=None, gui=False,scale=1.0, train=True, debug=False, epochs = 2, mem_size = 1500, decay=1-1e-4,
        batch_size = 80, sync_freq = 300, epsilon = 0.8, discount_factor=0.5, learning_rate = 1e-7, last_epoch=0, sumocfg_path="Simulation_Environment\Simulation Training\osm.sumocfg"):

    starting_epsilon = epsilon
    env = SumoEnvrionment(gui=gui, sumocfg_path=sumocfg_path, scale=scale)
    states_length = len(env.get_state(env.traffic_light))
    
    if train==True:
        net = Net(states_length,env.total_phases)
        target_net = Net(states_length,env.total_phases)
        
    else:
        if net != None:
            net = net
        else:
            print('No model in variable. Please add model')
            raise AttributeError

    replay = deque(maxlen=mem_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
    losses_progression=[]
    prev_sync = 0

    for i in tqdm(range(epochs)):
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
            
            # print(action)
            
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

        prev_sync = 0
        env.reset()

        if gui == False:
            traci.close()
            traci.start([env.sumoBinary, "-c", sumocfg_path,
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start", "--scale", f"{scale}"])
        if gui == True:
            traci.close()
            traci.load(["-c", sumocfg_path,
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start",  "--scale", f"{scale}"])

    target_net.to(torch.device('cpu'))
    local_time="_".join([str(i) for i in list(time.localtime())])

    cur_epoch = int(epochs) + int(last_epoch)
    save_path = f'DQN_Model/Model-{cur_epoch}epochs-_{local_time}'
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    

    #Saves Model and Losses Data
    torch.save(target_net, os.path.join('DQN_Model',Path(f'Model-{cur_epoch}epochs-_{local_time}.pth')))
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

    # plt.figtext(0.5,0.1," ".join(args_kwargs), ha="center", fontsize=10)
    plt.savefig(os.path.join(save_path, "Losses.png"))
    plt.show()

def evaluate(target_net, sumocfg_path="Simulation_Environment\Simulation Training\osm.sumocfg", scale=1.0):
    env = SumoEnvrionment(gui=True, sumocfg_path=sumocfg_path, evaluate=True, scale=scale)
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
    sumocfg_path2 = "Simulation_Environment\Main Route Simulation\osm.sumocfg"
    scale = 1.0
    sumocfg_path="Simulation_Environment\Simulation Training\osm.sumocfg"
    target_net, losses, save_path, args_kwargs = train(epochs=2, gui=False, sumocfg_path=sumocfg_path2, decay=0.99, scale=scale, learning_rate=1e-5)

    t1 = multiprocessing.Process(target=graph_losses, args=[losses, 0, save_path, args_kwargs])
    t2 = multiprocessing.Process(target=evaluate, args=[target_net, sumocfg_path2], kwargs={'scale':scale})

    t1.start()
    t2.start()

    t1.join()
    t2.join()