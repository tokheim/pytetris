from pytetris.ai import trainrunner
import argparse
import logging

parser = argparse.ArgumentParser('Python tetris game')
parser.add_argument('-n', '--name', type=str, default='test')
parser.add_argument('-r', '--restore', type=str, default=None)
parser.add_argument('-p', '--passive', action='store_true')
args = parser.parse_args()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

trainrunner.setup(args.name).run(args.restore)
