from setuptools import setup, find_packages

from pyMS import __version__

setup(name='pyMS',
      version=__version__,
      description='Python library for processing individual mass spectra',
      url='https://github.com/alexandrovteam/pyMS',
      author='Alexandrov Team, EMBL',
      packages=find_packages())
