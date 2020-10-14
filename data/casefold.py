#!/usr/bin/env python

import fileinput

import nltk


if __name__ == "__main__":
    for line in fileinput.input():
        print(line.rstrip().casefold())