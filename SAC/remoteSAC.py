from SAC.RL2023HockeyTournamentClient.client.backend import Client
from SAC.RL2023HockeyTournamentClient.client.remoteControllerInterface import RemoteControllerInterface
from SAC.sac import SACAgent


class RemoteSACAgent(SACAgent, RemoteControllerInterface):

    def __init__(self, **kwargs):
        SACAgent.__init__(self, **kwargs)
        RemoteControllerInterface.__init__(self, identifier='ATeamSAC')

    def remote_act(self, state):
        return self.select_action(state)


if __name__ == '__main__':
    controller = RemoteSACAgent()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='yourusername',
                    password='1234',
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