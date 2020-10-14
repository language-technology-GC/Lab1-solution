#!/usr/bin/env python
"""Computes Word2Vec embeddings for a list of target words."""


import argparse
import csv
import logging

import gensim  # type: ignore


def main(args: argparse.Namespace) -> None:
    # Trains w2v model.
    sentences = gensim.models.word2vec.LineSentence(args.tok_path)
    w2v = gensim.models.Word2Vec(
        sentences,
        iter=args.iter,
        min_count=args.min_count,
        size=args.size,
        window=args.window,
        workers=4,
    )
    # Computes cosine similarities for targeted pairs.
    with open(args.results_path, "w") as sink:
        writer = csv.writer(sink, delimiter="\t")
        with open(args.pairs_path) as source:
            for (x, y) in csv.reader(source, delimiter="\t"):
                try:
                    score = round(w2v.similarity(x, y), 6)
                    writer.writerow([x, y, score])
                except KeyError:
                    # This occurs when one of the words in the pair isn't
                    # in the vocabulary.
                    pass


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iter", default=5, help="number of iterations (default: %(default)s)"
    )
    parser.add_argument("--min_count", default=5, help="min count")
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
        "--size", default=100, help="embedding size (default: %(default)s)"
    )
    parser.add_argument(
        "--tok_path", required=True, help="path to input tokenized text file"
    )
    parser.add_argument(
        "--window",
        default=5,
        help="symmetric window size (default: %(default)s)",
    )
    main(parser.parse_args())
