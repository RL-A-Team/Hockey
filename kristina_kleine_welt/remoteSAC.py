from client import Client
from remoteControllerInterface import RemoteControllerInterface
from sac import SACAgent
from laserhockey import hockey_env as h_env


class RemoteSACAgent(SACAgent, RemoteControllerInterface):

    def __init__(self, **kwargs):
        SACAgent.__init__(self, **kwargs)
        RemoteControllerInterface.__init__(self, identifier='SAC')

    def remote_act(self, state):
        return self.select_action(state)


if __name__ == '__main__':
    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
    controller = RemoteSACAgent(state_dim=env.observation_space.shape, action_dim=env.action_space)
    

    # Play n (None for an infinite amount) games and quit
    client = Client(username='A Team', #'user0',
                    password= 'too5xeit3T', #'1234',
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
