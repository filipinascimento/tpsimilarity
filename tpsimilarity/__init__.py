# tpdistance/__init__.py
from tpsimilarity.similarity import *

from importlib.metadata import version

def get_version():
    return version("tpsimilarity")

__version__ = get_version()
