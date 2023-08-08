from client import Client
from remoteControllerInterface import RemoteControllerInterface
from sac import SACAgent
from laserhockey import hockey_env as h_env
import pickle


class RemoteSACAgent(SACAgent, RemoteControllerInterface):

    def __init__(self, loaded, **kwargs):
        self.loaded = loaded
        RemoteControllerInterface.__init__(self, identifier='SAC')

    def remote_act(self, state):
        print(self.loaded.select_action(state))
        return self.loaded.select_action(state)


if __name__ == '__main__':
    loaded = pickle.load(open('kristina_kleine_welt/sac_model_20230808T161743_22605.pkl', 'rb'))
    controller = RemoteSACAgent(loaded = loaded)
    

    # Play n (None for an infinite amount) games and quit
    client = Client(username='A Team',
                    password= 'too5xeit3T', #'too5xeit3T',
                    controller=controller,
                    output_path='logs/basic_opponents', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    # server_addr='localhost',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='user0',
    #                 password='1234',
    #                 controller=controller,
    #                 output_path='logs/basic_opponents',
    #                )
