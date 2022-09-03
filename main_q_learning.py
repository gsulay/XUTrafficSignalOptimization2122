import os
from secrets import token_urlsafe
import deep_q_test as dqn
from deep_q_test import Net, SumoEnvrionment
import torch

import os

# os.listdir(os.path.join(os.getcwd(), 'DQN_Model'))

def select_model():
    models_path = [i for i in os.listdir('DQN_Model') if i.split('.')[-1]=="pth"]
    text_display = "".join([f"{idx}:    {i} \n" for idx, i in enumerate(models_path)])

    while True:
        try:
            print("Select the model:")
            model_idx = int(input(text_display + "\n"))
            model = torch.load(os.path.join("DQN_Model",models_path[model_idx]))
            break
        except Exception as e:
            print(e)
    
    return model, models_path[model_idx]

if __name__ == '__main__':
    print("Select the run type:")
    while True:
        try:
            run_type = int(input("0:    New Model\n1:    Continue Train\n2:    Evaluate\n"))
            if run_type > 2:
                print("Invalid type. Please Try Again")
                raise ValueError
            
            if run_type == 1:
                model, path = select_model()
                epochs = int(input('How Many Epochs?:   '))
                last_epoch = path.split(r'\\')[-1].split('epochs')[0].split('-')[-1]
                last_epoch = int(last_epoch)

                print("Model Successfuly loaded from {}\nLast Epoch: ".format(path, last_epoch))
                print("Running SUMO...")
                print('-------------------------------------------------------------------')

                losses, epochs = dqn.train(model, last_epoch=last_epoch, epochs=epochs,sumocfg_path="Simulation_Environment\Main Route Simulation\osm.sumocfg" )
                
                print('Running Loss Graph...')

                dqn.graph_losses(losses, last_epoch)
            elif run_type == 2:
                model, path = select_model()
                scale = float(input("Scale Factor (default is 1): "))
                print("Model Successfuly loaded from {}".format(path))
                print("Running SUMO...")
                print('--------------------------------------------------------------------')
                dqn.evaluate(model, sumocfg_path="Simulation_Environment\Main Route Simulation\osm.sumocfg", scale=scale)
            elif run_type == 0:
                epochs = int(input('How Many Epochs?:   '))
                print("Running SUMO...")
                print('-------------------------------------------------------------------')
                target_net, losses = dqn.train(epochs=epochs, gui=False)
                dqn.evaluate(target_net)
                dqn.graph_losses(losses)
                
            break
        except Exception as e:
            print(e)
            print('Please Try Again....')
            
            
        


    
