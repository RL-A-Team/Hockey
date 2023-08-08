import numpy as np

from remoteControllerInterface import RemoteControllerInterface
from client import Client

class RemoteRandomOpponent(RemoteControllerInterface):

    def __init__(self, weak):
        RemoteControllerInterface.__init__(self, identifier='Randy Oppenheimer')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return np.random.uniform(-1,1,4)
        

if __name__ == '__main__':
    controller = RemoteRandomOpponent(weak=False)
    '''
    # Play n (None for an infinite amount) games and quit
    client = Client(username='stud54', # Testuser
                    password='too5xeit3T',
                    controller=controller, 
                    output_path='/tmp/ALRL2023/client/user0', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)
    '''               
    # Play n (None for an infinite amount) games and quit
    client = Client(username='A_Team',
                    password='too5xeit3T',
                    controller=controller,
                    output_path='logs/basic_opponents', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    #server_addr='localhost',#TODO comment out
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queuing
    # client = Client(username='user0', 
    #                 password='1234',
    #                 controller=controller, 
    #                 output_path='/tmp/ALRL2020/client/user0',
    #                )
