# Machine Learning for Juggling

A collection of experiments at the cross section of machine learning and
juggling of all things. The intent is for these to be well documented, cool, and
easily reproducible.

## Installation

jml is managed via [pipenv](https://github.com/kennethreitz/pipenv), a modern
python project management tool. It uses python 3.4.3+. 

This installation should be portable to any linux system using python3.4+.
Simply clone this repository, `pip3 install pipenv`, and then set up using
pipenv in this directory. All dependencies (including tensorflow, opencv, and
matplotlib) are portable wheels. This results in non-optimal performance, but
easy reproducibility.
