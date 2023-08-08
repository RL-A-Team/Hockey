# Client for the ALRL2020 Hockey Tournament

## RemoteController Interface

The client expects your Controller to be of type `RemoteControllerInterface`. You have to implement at least the abstract method `remote_act` provided by the interface. See `remoteBasicOpponent.py` for a reference implementation.

## Client

The client can be run in one of two modes: `interactive` and `non-interactive`.

### Interactive Mode

In interactive mode, you can interact with the client via a `cmd`-interface. As soon as you start queueing for games (`start_queueing`) you will be continuously paired with other players to play against each other. You can enter the `cmd`-interface while playing/queuing by pressing `<ESC>`.

### Non-Interactive Mode

You can run the client in the non-interactive mode by setting `interactive = False` and `op = 'start_queueing'`.

If `num_games = None` the client will continuously queuing for new games. You can limit the number of games played by setting `num_games : int > 0`.

## Train loop

If you want to use games played online to train your policy, we suggest to run one worker process with the client in non-interactive mode to collect data and one process that trains the policy. 

The train process has to load new data from the disc and add it to the replay buffer while the data collection worker should update the policy parameters from time to time.

