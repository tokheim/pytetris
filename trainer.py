from pytetris.ai import trainrunner
import argparse
import logging
import shutil

parser = argparse.ArgumentParser('Python tetris game')
parser.add_argument('-n', '--name', type=str, default='test')
parser.add_argument('--score_restore', type=str, default=None)
parser.add_argument('--board_restore', type=str, default=None)
parser.add_argument('-s', '--noscreen', action='store_true')
parser.add_argument('-i', '--imagedir', type=str, default="images/")
parser.add_argument('--train_board', action='store_true')
parser.add_argument('--train_score', action='store_true')
args = parser.parse_args()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

shutil.rmtree('summary/'+args.name, ignore_errors=True)
draw_every = 100
if not (args.train_board or args.train_score):
    log.warn("no training target!")
if args.noscreen:
    draw_every = None
runner = trainrunner.setup(args.name, draw_every, args.imagedir, args.train_board, args.train_score)
runner.run(args.board_restore, args.score_restore)
