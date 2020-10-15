#!/usr/bin/env python

import fileinput

import nltk  # type: ignore


if __name__ == "__main__":
    for line in fileinput.input():
        print(" ".join(nltk.word_tokenize(line)))
