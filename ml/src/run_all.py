"""Run the whole benchmark end-to-end and produce the comparison chart."""

from __future__ import annotations

import argparse

import evaluation
import model1_naive_bayes
import model2_tfidf_linear


def main(dataset: str, sample: int | None) -> None:
    model1_naive_bayes.main(dataset, sample)
    model2_tfidf_linear.main(dataset, sample, "logreg")
    model2_tfidf_linear.main(dataset, sample, "svm")
    evaluation.plot_comparison()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cornell", choices=["cornell", "imdb", "crawled"])
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    main(args.dataset, args.sample)
