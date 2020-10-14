#!/usr/bin/env python
"""Computes (P)PMI table for a list of target word pairs."""


import argparse
import csv
import collections
import logging
import math

from typing import Counter


def _pmi(pxy: float, px: float, py: float) -> float:
    return math.log2(pxy) - math.log2(px) - math.log2(py)


def _ppmi(pxy: float, px: float, py: float) -> float:
    return max(_pmi(pxy, px, py), 0.0)


def main(args: argparse.Namespace) -> None:
    # Reads target words and pairs from TSV file.
    wordlist = []
    pairlist = []
    with open(args.pairs_path) as source:
        for pair in csv.reader(source, delimiter="\t"):
            pair[0] = pair[0].casefold()
            pair[1] = pair[1].casefold()
            pair.sort()
            wordlist.extend(pair)
            pairlist.append(tuple(pair))
    words = frozenset(wordlist)
    pairs = frozenset(pairlist)
    logging.info("%d words tracked", len(words))
    logging.info("%d pairs tracked", len(pairs))
    # Reads counts from the data.
    nwords = 0
    npairs = 0
    cwords: Counter[str] = collections.Counter()
    cpairs: Counter[str] = collections.Counter()
    with open(args.tok_path, "r") as source:
        for line in source:
            tokens = line.rstrip().casefold().split()
            for (i, x) in enumerate(tokens):
                nwords += 1
                left_window = tokens[i - args.window:i]
                right_window = tokens[i + 1:i + 1 + args.window]
                window = left_window + right_window
                npairs += len(window)
                if x not in wordlist:
                    continue
                for y in window:
                    if y not in wordlist:
                        continue
                    pair = [x, y]
                    pair.sort()
                    pair = tuple(pair)  # type: ignore
                    if pair not in pairlist:
                        continue
                    cwords[x] += 1
                    cwords[y] += 1
                    cpairs[pair] += 1  # type: ignore
    logging.info("%d tokens counted", nwords)
    logging.info("%d pairs covered", len(cpairs))
    # Computes and writes out (P)PMI scores.
    scorer = _pmi if args.pmi else _ppmi
    with open(args.results_path, "w") as sink:
        writer = csv.writer(sink, delimiter="\t")
        for (pair, cpair) in cpairs.items():  # type: ignore
            pxy = cpair / npairs
            (x, y) = pair
            px = cwords[x] / nwords
            py = cwords[y] / nwords
            score = round(scorer(pxy, px, py), 6)
            writer.writerow([x, y, score])


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_path",
        required=True,
        help="path to output three-column TSV file containing PPMI results",
    )
    parser.add_argument(
        "--pairs_path",
        required=True,
        help="path to input two-column TSV file containing the "
        "target word pairs",
    )
    parser.add_argument(
        "--pmi",
        default=False,
        action="store_true",
        help="compute PMI instead of PPMI",
    )
    parser.add_argument(
        "--tok_path", required=True, help="path to input tokenized text file"
    )
    parser.add_argument(
        "--window",
        default=10,
        help="symmetric window size (default: %(default)s)",
    )
    main(parser.parse_args())
