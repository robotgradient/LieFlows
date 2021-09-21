from setuptools import setup
from codecs import open
from os import path


from liesvf import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='liesvf',
      version=__version__,
      description='Lie Stable Vector Fields',
      author='Julen Urain',
      author_email='julen@robot-learning.de',
      packages=['liesvf'],
      install_requires=requires_list,
      )
