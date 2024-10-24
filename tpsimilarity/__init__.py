# tpdistance/__init__.py
from tpsimilarity.similarity import *

import toml
from pathlib import Path

def get_version():
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
    with open(pyproject_path, 'r') as f:
        pyproject_data = toml.load(f)
    return pyproject_data['project']['version']

__version__ = get_version()
