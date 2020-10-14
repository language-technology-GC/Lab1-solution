#!/usr/bin/env python
"""Computes correlations for WordNet similarity measures."""

import logging

from typing import List, Optional

from nltk.corpus import wordnet, wordnet_ic
from nltk.corpus.reader.wordnet import Synset, WordNetError
import pandas
import scipy.stats


# The combined scores from the WordSimilarity-353 Test Collection:
#
#     http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/
HUMAN_SOURCE = "data/ws353.tsv"

NAN = float("nan")


class Error(Exception):

    pass


class SimilarityCalculator:
    """Computes similarity tuples for pairs of strings.

    This is an attempt to hack around some of the deficiencies of the
    NLTK WordNet API. To wit:

    * There's no default logic for choosing the "best" synset when there are
      multiple options.
    * The similarity functions are inexplicably methods of the Synset object
      rather than functions which take two words or two Synsets.
    * There's no way of binding an information content object to the
      similarity functions that need it.
    """

    # TODO: Add support for other ways to build an IC object.

    def __init__(self, ic_path: str = "ic-brown.dat"):
        self.brown_ic = wordnet_ic.ic(ic_path)

    @staticmethod
    def synset(
        lemma: str, pos: Optional[str] = None, lang: str = "eng"
    ) -> Synset:
        synsets = wordnet.synsets(lemma, pos, lang)
        if not synsets:
            raise Error(f"No synsets found for {lemma}")
        # I choose the simple strategy of selecting the first (supposedly, most
        # frequent) synset.
        return synsets[0]

    # The actual similarity functions.

    def path(self, s1: Synset, s2: Synset) -> float:
        return s1.path_similarity(s2)

    def lch(self, s1: Synset, s2: Synset) -> float:
        try:
            return s1.lch_similarity(s2)
        except WordNetError:
            return NAN

    def wup(self, s1: Synset, s2: Synset) -> float:
        return s1.wup_similarity(s2)

    def res(self, s1: Synset, s2: Synset) -> float:
        try:
            return s1.res_similarity(s2, self.brown_ic)
        except WordNetError:
            return NAN

    def jcn(self, s1: Synset, s2: Synset) -> float:
        try:
            return s1.jcn_similarity(s2, self.brown_ic)
        except WordNetError:
            return NAN

    def lin(self, s1: Synset, s2: Synset) -> float:
        try:
            return s1.lin_similarity(s2, self.brown_ic)
        except WordNetError:
            return NAN


def _cor(x, y) -> float:
    """Computes Spearman correlation coefficient."""
    return scipy.stats.spearmanr(x, y, nan_policy="omit").correlation


def main() -> None:
    # Reads in human similarity data.
    data = pandas.read_csv(HUMAN_SOURCE, delimiter="\t",
                           names=["x", "y", "sim"])
    # Casefolds.
    data["x"] = data["x"].str.casefold()
    data["y"] = data["y"].str.casefold()
    # Grabs synset pairs.
    synset_pairs: List[tuple[Synset, Synset]] = []
    for (_, w1, w2, score) in data.itertuples():
        s1 = SimilarityCalculator.synset(w1)
        s2 = SimilarityCalculator.synset(w2)
        synset_pairs.append((s1, s2))
    # Adds similarity scores.
    simcalc = SimilarityCalculator()
    data["path"] = [simcalc.path(s1, s2) for (s1, s2) in synset_pairs]
    data["lch"] = [simcalc.lch(s1, s2) for (s1, s2) in synset_pairs]
    data["wup"] = [simcalc.wup(s1, s2) for (s1, s2) in synset_pairs]
    data["res"] = [simcalc.res(s1, s2) for (s1, s2) in synset_pairs]
    data["jcn"] = [simcalc.jcn(s1, s2) for (s1, s2) in synset_pairs]
    data["lin"] = [simcalc.lin(s1, s2) for (s1, s2) in synset_pairs]
    # Computes correlations.
    logging.info("path:\t%.4f", _cor(data["sim"], data["path"]))
    logging.info("lch:\t%.4f", _cor(data["sim"], data["lch"]))
    logging.info("wup:\t%.4f", _cor(data["sim"], data["wup"]))
    logging.info("res:\t%.4f", _cor(data["sim"], data["res"]))
    logging.info("jcn:\t%.4f", _cor(data["sim"], data["jcn"]))
    logging.info("lin:\t%.4f", _cor(data["sim"], data["lin"]))


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    main()
