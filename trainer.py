from pytetris.ai import trainrunner
import argparse
import logging
import shutil

parser = argparse.ArgumentParser('Python tetris game')
parser.add_argument('-n', '--name', type=str, default='test')
parser.add_argument('-r', '--restore', type=str, default=None)
parser.add_argument('-s', '--noscreen', action='store_true')
args = parser.parse_args()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

shutil.rmtree('summary/'+args.name, ignore_errors=True)
if not args.noscreen:
    trainrunner.setup(args.name).run(args.restore)
else:
    trainrunner.setup(args.name, None).run(args.restore)
