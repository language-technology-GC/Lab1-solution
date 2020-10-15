#!/usr/bin/env python
"""Computes correlation and coverage results."""

import argparse
import csv
import logging

from typing import Dict, List, Tuple

import scipy.stats  # type: ignore


def _make_pair(x: str, y: str) -> Tuple[str, str]:
    pair = [x.casefold(), y.casefold()]
    pair.sort()
    return tuple(pair)  # type: ignore


def main(args: argparse.Namespace) -> None:
    # Reads in human similarity data.
    with open(args.ws353_path, "r") as source:
        ws353: Dict[Tuple[str, str], float] = {}
        for (x, y, score) in csv.reader(source, delimiter="\t"):
            ws353[_make_pair(x, y)] = float(score)
    # Reads the results data.
    ws_score: List[float] = []
    rs_score: List[float] = []
    with open(args.results_path, "r") as source:
        for (x, y, score) in csv.reader(source, delimiter="\t"):
            ws_score.append(ws353[_make_pair(x, y)])
            rs_score.append(float(score))
    # Computes statistics.
    rho = scipy.stats.spearmanr(ws_score, rs_score).correlation
    coverage = len(rs_score) / len(ws353)
    logging.info("Cor:\t% .4f (coverage: %.2f)", rho, coverage)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ws353_path", required=True, help="path to ws353 TSV"
    )
    parser.add_argument(
        "--results_path", required=True, help="path to results TSV"
    )
    main(parser.parse_args())
