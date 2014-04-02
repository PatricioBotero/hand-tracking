import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Hand Tracking",
    version = "0.0.1",
    author = "Patrick Buchter",
    author_email = "patrick.buchter@gmail.com",
    description = ("tracking a hand with particle filter "
                   "and the bhattacharyya distance as likelihood function."),
    license = "BSD",
    keywords = "tracking, particle filter, webcam",
    url = "",
    packages=['src', 'test'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)