import argparse
from pytetris import gameengine

parser = argparse.ArgumentParser('Python tetris game')
parser.add_argument('-x', '--width', default=10, type=int)
parser.add_argument('-y', '--height', default=20, type=int)
parser.add_argument('-b', '--blocksize', default=30, type=int)

args = parser.parse_args()

ge = gameengine.create_game(args.width, args.height, args.blocksize)
