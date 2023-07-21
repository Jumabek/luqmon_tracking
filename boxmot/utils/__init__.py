from loguru import logger
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
BOXMOT = ROOT / 'boxmot'
EXAMPLES = ROOT / 'examples'
WEIGHTS = ROOT / 'examples' / 'weights'
REQUIREMENTS = ROOT / 'requirements.txt'

# global logger
logger.remove()
logger.add(sys.stderr, colorize=True)

variables = {
    'FILE': FILE,
    'ROOT': ROOT,
    'BOXMOT': BOXMOT,
    'EXAMPLES': EXAMPLES,
    'WEIGHTS': WEIGHTS,
    'REQUIREMENTS': REQUIREMENTS
}

for name, value in variables.items():
    print(f'{name}: {value}')
