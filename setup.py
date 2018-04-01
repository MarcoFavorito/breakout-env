import os
import re
from setuptools import setup, find_packages

try:
  from pypandoc import convert
  read_md = lambda f: convert(f, 'rst')
except ImportError:
  print("warning: pypandoc module not found, could not convert Markdown to RST")
  read_md = lambda f: open(f, 'r').read()

def read_version():
  # __PATH__ = os.path.abspath(os.path.dirname(__file__))
  # with open(os.path.join(__PATH__, 'breakout_env/__init__.py')) as f:
  #   version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
  # if version_match:
  #   return version_match.group(1)
  # raise RuntimeError("Unable to find __version__ string")
  return "1.0.5"

setup(
  name='breakout_env',
  packages=find_packages(include=['breakout_env*']),
  version=read_version(),
  description='A configurable Breakout environment for reinforcement learning',
  long_description=read_md('README.md'),
  author='SSARCandy',
  author_email='ssarcandy@gmail.com',
  license='MIT',
  url='https://github.com/SSARCandy/breakout-env',
  keywords=['game', 'learning', 'evironment'],
  classifiers=[],
  install_requires=['numpy>=1.1', 'distribute'],
  include_package_data=True
)