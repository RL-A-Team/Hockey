import argparse
import ast
import json
import time

import imageio
from PIL import Image
import numpy as np
import os
from glob import glob
import datetime

from laserhockey.hockey_env import HockeyEnv, FPS, CENTER_X, CENTER_Y
from numpy.lib.function_base import trim_zeros

def set_env_state_from_observation(env, observation):
    env.player1.position = (observation[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
    env.player1.angle = observation[2]
    env.player1.linearVelocity = [observation[3], observation[4]]
    env.player1.angularVelocity = observation[5]
    env.player2.position = (observation[[6, 7]] + [CENTER_X, CENTER_Y]).tolist()
    env.player2.angle = observation[8]
    env.player2.linearVelocity = [observation[9], observation[10]]
    env.player2.angularVelocity = observation[11]
    env.puck.position = (observation[[12, 13]] + [CENTER_X, CENTER_Y]).tolist()
    env.puck.linearVelocity = [observation[14], observation[15]]

def setup_video(output_path, id, fps):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{id}.mp4")
    print("Record video in {}".format(file_path))
    # noinspection SpellCheckingInspection
    return (
        imageio.get_writer(file_path, fps=fps, codec="mjpeg", quality=10, pixelformat="yuvj444p"),
        file_path)


def main(games_path, id, record, render, output_path,
         verbose):

    env = HockeyEnv()

    matches = [np.load(match, allow_pickle=True)['arr_0'].item() for match in glob(os.path.join(games_path, '**', '*.npz'), recursive=True)]
    if not id is None:
        matches = [match for match in matches if match['identifier'] == id]

    for match in matches:
        if verbose:
            print('Match id: ', match['identifier'])
            print('Date:' + datetime.date.fromtimestamp(match['timestamp']).strftime("%m/%d/%Y, %H:%M:%S"))
            print(f'{match["player_one"]} vs {match["player_two"]}')
        # noinspection PyChainedComparisons

        if record:
            video, video_path = setup_video(output_path, match['identifier'], FPS)

        for transition in match['transitions']:
            set_env_state_from_observation(env, np.asfarray(transition[0]))

            if verbose:
                if transition[4] or transition[5]:  # done or truncated
                    if transition[6]['winner'] == 0:
                        print(f'Game end in a draw')
                    elif transition[6]['winner'] == 1:
                        print(f'{match["player_one"]} scored.')
                    else:
                        print(f'{match["player_two"]} scored.')

            if record:
                frame = env.render(mode="rgb_array")
                # noinspection PyUnboundLocalVariable
                video.append_data(frame)
            elif render:
                env.render()
                time.sleep(1/FPS)

        if record:
            video.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-path', help='Path to games')
    parser.add_argument('--record', action='store_true', help='Whether to record video or not')
    parser.add_argument('--render', action='store_true', help='Whether to render in realtime or not')
    parser.add_argument('--id', default=None, help='id of game you want to replay, do not specify to run all')
    parser.add_argument('--output-path', default=None, help='Where to save video')
    parser.add_argument('--verbose', action='store_true', help='Print more info')

    args = parser.parse_args()
    main(args.games_path, args.id, args.record, args.render, args.output_path,
         args.verbose)
