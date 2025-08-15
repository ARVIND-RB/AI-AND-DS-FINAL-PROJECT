AI AND DS FINAL PROJECT — FINAL-Projects
Demos:
  A) Early‑Risk Prediction (Logistic Regression)
  B) Content‑Based Recommender (TF‑IDF + Cosine Similarity)
  C) Short‑Answer Grading Helper (TF‑IDF + Keywords)
"""
import argparse
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Args:
    demo: str = "risk"
    save_plots: bool = False
    topn: int = 3


def demo_risk(save_plots: bool = False) -> None:
    """
    A) Early‑Risk Prediction — Logistic Regression
    Predict if a student is at risk using engagement + quiz features.
    """
    np.random.seed(42)
    N = 800
    attendance = np.clip(np.random.normal(0.8, 0.15, N), 0, 1)            # 0-1
    hours_on_platform = np.random.gamma(shape=2.0, scale=2.0, size=N)      # hours
    quiz_avg = np.clip(np.random.normal(0.7, 0.18, N), 0, 1)               # 0-1
    late_submissions = np.random.poisson(lam=1.2, size=N)                  # count
    forum_posts = np.random.poisson(lam=3.0, size=N)

    # Target: 1 = at risk
    logit = -2.2 + (-2.5*attendance) + (-1.4*quiz_avg) + (0.25*late_submissions) + (-0.05*forum_posts)
    prob = 1/(1+np.exp(-logit))
    y = (np.random.rand(N) < prob).astype(int)

    X = pd.DataFrame({
        'attendance': attendance,
        'hours_on_platform': hours_on_platform,
        'quiz_avg': quiz_avg,
        'late_submissions': late_submissions,
        'forum_posts': forum_posts,
    })

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print("=== Demo A: Early‑Risk Prediction ===")
    print(classification_report(y_test, pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

    coef = pd.Series(clf.coef_[0], index=X.columns).sort_values()
    print("\nFeature effects (log-odds):")
    print(coef)

    # Save a quick horizontal bar plot (no explicit colors)
    if save_plots:
        ax = coef.plot(kind="barh", title="Feature Effects (log-odds)")
        plt.tight_layout()
        out = Path("feature_effects_risk.png")
        plt.savefig(out)
        plt.close()
        print(f"\nSaved plot: {out.resolve()}")


def demo_recommender(topn: int = 3) -> None:
    """
    B) Content‑Based Recommender — TF‑IDF + Cosine Similarity
    Recommend learning resources based on student profile text.
    """
    resources = pd.DataFrame([
        {"id": 1, "title": "Intro to Linear Regression", "desc": "supervised learning regression MSE gradient descent"},
        {"id": 2, "title": "Classification with Logistic Regression", "desc": "binary classification probability sigmoid regularization"},
        {"id": 3, "title": "Decision Trees and Random Forests", "desc": "bagging trees interpretability feature importance"},
        {"id": 4, "title": "K-Means Clustering", "desc": "unsupervised learning clusters centroids inertia"},
        {"id": 5, "title": "PCA for Dimensionality Reduction", "desc": "eigenvectors variance projection SVD"},
        {"id": 6, "title": "Model Evaluation", "desc": "precision recall ROC AUC cross-validation overfitting"},
    ])

    # Example student profile: could come from interests, clicked topics, recent quiz tags
    student_profile = "classification probability ROC recall precision model evaluation"

    corpus = resources["desc"].tolist() + [student_profile]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)
    resource_vecs, student_vec = X[:-1], X[-1]
    sims = cosine_similarity(resource_vecs, student_vec)
    resources["score"] = sims

    top = resources.sort_values("score", ascending=False).head(topn)
    print("\n=== Demo B: Top recommendations ===")
    print(top[["id", "title", "score"]].to_string(index=False))


def demo_grading() -> None:
    """
    C) Short‑Answer Grading Helper — TF‑IDF + Keywords (HITL)
    Assists teachers by scoring alignment to a reference answer and rubric keywords.
    """
    prompt = "Explain the bias-variance tradeoff in model selection."
    reference = (
        "Bias-variance tradeoff balances underfitting and overfitting. "
        "High bias models underfit with systematic error; high variance models overfit and are sensitive to noise. "
        "Optimal complexity minimizes expected generalization error."
    )
    rubric_keywords = ["underfitting", "overfitting", "complexity", "generalization", "noise", "systematic error", "sensitivity"]

    answers = pd.Series([
        "Models with high bias are too simple and underfit; high variance models overfit and change a lot with new data. We need a balance.",
        "It is about gradients and learning rate tuning to reduce loss quickly.",
        "Bias is constant error, variance is fluctuation due to noise; pick model complexity to reduce total error.",
    ])

    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(list(answers) + [reference])
    A, R = X[:-1], X[-1]
    content_scores = cosine_similarity(A, R)[:, 0]  # 0-1

    kw = [k.lower() for k in rubric_keywords]

    def coverage(text: str) -> float:
        t = re.sub(r"[^a-zA-Z ]", " ", text).lower()
        tokens = set(t.split())
        hits = sum(any(w in tokens for w in k.split()) for k in kw)
        return hits / len(kw)

    coverage_scores = answers.apply(coverage)
    final = 0.7 * content_scores + 0.3 * coverage_scores

    result = pd.DataFrame({
        "answer": answers,
        "similarity_to_reference": np.round(content_scores, 3),
        "rubric_coverage": np.round(coverage_scores, 3),
        "combined_score": np.round(final, 3),
    }).sort_values("combined_score", ascending=False)

    print("\n=== Demo C: Grading Helper ===")
    print(f"Prompt: {prompt}\n")
    print("Reference:", reference, "\n")
    print(result.to_string(index=False))
    print("\nNote: This is a drafting aid. A teacher should review borderline cases before assigning final grades.")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="AI in Education — Mini-Projects")
    parser.add_argument("--demo", choices=["risk", "recommender", "grading"], default="risk",
                        help="Which demo to run")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to PNG (risk demo)")
    parser.add_argument("--topn", type=int, default=3, help="Top-N recommendations (recommender demo)")
    ns = parser.parse_args()
    return Args(demo=ns.demo, save_plots=ns.save_plots, topn=ns.topn)


def main() -> None:
    args = parse_args()
    if args.demo == "risk":
        demo_risk(save_plots=args.save_plots)
    elif args.demo == "recommender":
        demo_recommender(topn=args.topn)
    elif args.demo == "grading":
        demo_grading()
    else:
        raise ValueError("Unknown demo")


if __name__ == "__main__":
    main()
