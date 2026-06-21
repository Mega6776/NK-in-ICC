import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# File locations and modeling settings used throughout the pipeline.
DATA_DIR = Path(r"data/processed")
OUTPUT_DIR = DATA_DIR / "cibersortDetails"
EXPRESSION_FILE = "combined_gene_level.xlsx"
CIBERSORT_FILE = "combined_cibersort_output.xlsx"
ACTIVATED_NK_COLUMN = "NK cells activated"
RESTING_NK_COLUMN = "NK cells resting"
TEST_SIZE = 0.2
OUTER_FOLDS = 5
INNER_FOLDS = 5
VARIANCE_PERCENTILE = 10
K_BEST_FEATURES = 500
RFE_FEATURES = 100
L1_RATIOS = np.linspace(0.1, 0.9, 9)
REGULARIZATION_STRENGTHS = [0.01, 0.1, 1, 10]


# Custom SelectKBest wrapper.
class AdaptiveSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k=500, score_func=f_classif):
        self.k = k
        self.score_func = score_func

    def fit(self, features, labels=None):
        effective_k = min(self.k, features.shape[1])
        self.selector_ = SelectKBest(score_func=self.score_func, k=effective_k)
        fitted_labels = labels if labels is not None else np.zeros(features.shape[0], dtype=int)
        self.selector_.fit(features, fitted_labels)
        self.support_ = self.selector_.get_support()
        return self

    def transform(self, features):
        return self.selector_.transform(features)

    def get_support(self, indices=False):
        return self.selector_.get_support(indices=indices)


# Custom RFE wrapper.
class AdaptiveRFE(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, n_features_to_select=100, step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step

    def fit(self, features, labels):
        # RFE ranks genes by repeatedly fitting this base logistic-regression estimator.
        base_estimator = self.estimator or LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=10000,
        )
        effective_feature_count = min(self.n_features_to_select, features.shape[1])
        self.rfe_ = RFE(
            estimator=clone(base_estimator),
            n_features_to_select=effective_feature_count,
            step=self.step,
        )
        self.rfe_.fit(features, labels)
        self.support_ = self.rfe_.get_support()
        return self

    def transform(self, features):
        return self.rfe_.transform(features)

    def get_support(self, indices=False):
        return self.rfe_.get_support(indices=indices)


# Create the output folder
def configure_environment():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")


# Match a required column name
def resolve_column_name(columns, requested_name):
    if requested_name in columns:
        return requested_name
    names_by_lowercase = {column.lower(): column for column in columns}
    resolved_name = names_by_lowercase.get(requested_name.lower())
    if resolved_name is None:
        raise KeyError(f"Could not find required column: {requested_name}")
    return resolved_name


# Load gene-expression data and transpose it into samples x genes format.
def load_expression_matrix():
    expression_frame = pd.read_excel(DATA_DIR / EXPRESSION_FILE, index_col=0)
    sample_by_gene = expression_frame.T
    print(f"Expression matrix shape (samples x genes): {sample_by_gene.shape}")
    return sample_by_gene


# Load CIBERSORT NK-cell fractions and convert them into binary labels.
def load_nk_labels():
    counts_frame = pd.read_excel(DATA_DIR / CIBERSORT_FILE, header=0)
    if "Mixture" not in counts_frame.columns:
        raise KeyError("Expected a 'Mixture' column containing sample IDs.")

    counts_frame["Mixture"] = counts_frame["Mixture"].astype(str).str.strip()
    counts_frame = counts_frame.set_index("Mixture")

    activated_column = resolve_column_name(counts_frame.columns, ACTIVATED_NK_COLUMN)
    resting_column = resolve_column_name(counts_frame.columns, RESTING_NK_COLUMN)

    activated_values = pd.to_numeric(counts_frame[activated_column], errors="coerce")
    resting_values = pd.to_numeric(counts_frame[resting_column], errors="coerce")
    # Positive difference means activated NK fraction is at least resting NK fraction.
    counts_frame["nk_difference"] = activated_values - resting_values
    # Label 1 = activated NK >= resting NK; label 0 = activated NK < resting NK.
    counts_frame["nk_label"] = (counts_frame["nk_difference"] >= 0).astype(int)
    return counts_frame["nk_label"]


# Keep only samples that appear in both the expression matrix and the NK-label table.
def align_expression_and_labels(expression_matrix, labels):
    cleaned_sample_ids = pd.Index(expression_matrix.index.astype(str).str.strip())
    aligned_labels = labels.reindex(cleaned_sample_ids)
    matched_sample_count = int(aligned_labels.notna().sum())

    print(
        f"Matched {matched_sample_count} / {len(cleaned_sample_ids)} samples "
        "between expression matrix and counts file."
    )

    if matched_sample_count == 0:
        raise ValueError("No sample IDs matched between expression and counts files.")

    aligned_labels.index = expression_matrix.index
    labeled_sample_mask = aligned_labels.notna()
    labeled_expression = expression_matrix.loc[labeled_sample_mask.index[labeled_sample_mask]].copy()
    labeled_labels = aligned_labels.loc[labeled_sample_mask.index[labeled_sample_mask]].astype(int)

    print(
        f"Using {labeled_expression.shape[0]} labeled samples for modeling. "
        f"Label counts:\n{labeled_labels.value_counts()}"
    )
    return labeled_expression, labeled_labels


# Split labeled samples into train and held-out test sets.
# random_state=None
def split_train_test(expression_matrix, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(
        expression_matrix,
        labels,
        test_size=TEST_SIZE,
        # Preserve approximately the same class balance in train and test sets.
        stratify=labels,
        random_state=None,
    )
    print(f"Train samples: {train_features.shape[0]}, Test samples: {test_features.shape[0]}")
    return train_features, test_features, train_labels, test_labels


# Elastic-net logistic regression with inner cross-validation for C and l1_ratio.
def make_logistic_cv():
    return LogisticRegressionCV(
        # Elastic net combines L1-style sparsity with L2-style stability.
        penalty="elasticnet",
        solver="saga",
        l1_ratios=L1_RATIOS,
        Cs=REGULARIZATION_STRENGTHS,
        cv=INNER_FOLDS,
        # Tune hyperparameters by ROC AUC instead of raw accuracy.
        scoring="roc_auc",
        max_iter=10000,
        n_jobs=-1,
        refit=True,
    )


# PCA comparison pipeline: all genes -> scaling -> PCA -> elastic-net logistic regression.
def make_pca_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("classifier", make_logistic_cv()),
        ]
    )


# Gene-selection pipeline: variance filter -> scaling -> ANOVA KBest -> RFE -> classifier.
def make_gene_pipeline(variance_threshold):
    rfe_estimator = LogisticRegression(penalty="l2", solver="liblinear", max_iter=10000)
    return Pipeline(
        [
            # Remove low-variance genes
            ("variance_filter", VarianceThreshold(threshold=variance_threshold)),
            ("scaler", StandardScaler()),
            # Keep genes with the strongest ANOVA relationship to the labels.
            ("k_best", AdaptiveSelectKBest(k=K_BEST_FEATURES, score_func=f_classif)),
            (
                "rfe",
                AdaptiveRFE(
                    estimator=rfe_estimator,
                    n_features_to_select=RFE_FEATURES,
                    step=1,
                ),
            ),
            ("classifier", make_logistic_cv()),
        ]
    )


# Run outer cross-validation to estimate performance on unseen folds.
def cross_validate_pipeline(pipeline, features, labels, pipeline_name):
    # random_state=None
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=None)
    print(f"\nPerforming cross-validation for {pipeline_name}...")
    auc_scores = cross_val_score(
        pipeline,
        features.values,
        labels,
        # Tune hyperparameters by ROC AUC instead of raw accuracy.
        scoring="roc_auc",
        cv=outer_cv,
        n_jobs=-1,
    )
    accuracy_scores = cross_val_score(
        pipeline,
        features.values,
        labels,
        scoring="accuracy",
        cv=outer_cv,
        n_jobs=-1,
    )
    print(f"{pipeline_name} CV ROC AUC mean ± std: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(
        f"{pipeline_name} CV Accuracy mean ± std: "
        f"{accuracy_scores.mean():.3f} ± {accuracy_scores.std():.3f}"
    )
    return auc_scores, accuracy_scores


def evaluate_classifier(labels, predictions, probabilities):
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "cohen_kappa": cohen_kappa_score(labels, predictions),
        "avg_precision": average_precision_score(labels, probabilities)
        if len(np.unique(labels)) > 1
        else np.nan,
        "roc_auc": roc_auc_score(labels, probabilities) if len(np.unique(labels)) > 1 else np.nan,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, zero_division=0),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    return metrics


# Print a readable metric block.
def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for metric_name, metric_value in metrics.items():
        if metric_name in {"confusion_matrix", "classification_report"}:
            print(f"{metric_name}:\n{metric_value}")
        else:
            print(f"{metric_name}: {metric_value}")


def plot_variance_histograms(train_features, variance_threshold):
    # Calculate the variance threshold from training data
    gene_variances = train_features.astype(float).var(axis=0, ddof=1)

    plt.figure(figsize=(8, 4))
    plt.hist(gene_variances, bins=100, alpha=0.9)
    plt.axvline(
        variance_threshold,
        color="red",
        linestyle="--",
        label=f"threshold={variance_threshold:.3g}",
    )
    plt.xlabel("Raw gene variance (training)")
    plt.ylabel("Count")
    plt.title("Per-gene variance (raw training)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"variance_raw_hist_trial.png", dpi=300)
    plt.close()

    variance_filter = VarianceThreshold(threshold=variance_threshold).fit(train_features.values)

    plt.figure(figsize=(6, 4))
    plt.hist(gene_variances[variance_filter.get_support()], bins=60)
    plt.title("Per-gene variance (survivors after VT)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"variance_after_vt_hist_trial.png", dpi=300)
    plt.close()


def make_report_writer(report_path):
    def write_line(value):
        print(value)
        with open(report_path, "a", encoding="utf-8") as report_file:
            report_file.write(str(value) + "\n")

    return write_line


# Recover the gene names that survive each feature-selection step.
def get_selected_gene_sets(gene_pipeline, gene_names):
    variance_mask = gene_pipeline.named_steps["variance_filter"].get_support()
    genes_after_variance = gene_names[variance_mask]

    k_best_step = gene_pipeline.named_steps["k_best"]
    k_best_mask = k_best_step.get_support()
    genes_after_k_best = genes_after_variance[k_best_mask]

    rfe_step = gene_pipeline.named_steps["rfe"]
    rfe_mask = rfe_step.get_support()
    genes_after_rfe = genes_after_k_best[rfe_mask]

    return genes_after_variance, genes_after_k_best, genes_after_rfe


# Write a report describing the fitted feature-selection pipeline.
def write_feature_selection_report(gene_pipeline, train_features):
    report_path = OUTPUT_DIR / f"rfe_report_trial.txt"
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"RFE / feature-selection report - trial \n")
        report_file.write("Generated: " + time.asctime() + "\n\n")

    write_line = make_report_writer(report_path)
    gene_names = train_features.columns.to_numpy()
    genes_after_variance, genes_after_k_best, genes_after_rfe = get_selected_gene_sets(
        gene_pipeline,
        gene_names,
    )

    write_line("=== Pipeline components ===")
    write_line("Pipeline steps: " + " -> ".join(name for name, _ in gene_pipeline.steps))
    write_line("Pipeline representation:\n" + str(gene_pipeline))

    scaler = gene_pipeline.named_steps["scaler"]
    write_line("\n--- StandardScaler ---")
    write_line(f"mean first 10: {np.round(scaler.mean_[:10], 6).tolist()}")
    write_line(f"scale first 10: {np.round(scaler.scale_[:10], 6).tolist()}")

    variance_filter = gene_pipeline.named_steps["variance_filter"]
    variance_mask = variance_filter.get_support()
    retained_count = int(variance_mask.sum())
    removed_count = train_features.shape[1] - retained_count
    write_line("\n--- VarianceThreshold ---")
    write_line(f"removed {removed_count} features; retained {retained_count}")
    write_line("first 10 retained genes: " + ", ".join(list(genes_after_variance[:10])))

    write_anova_report(gene_pipeline, genes_after_variance, write_line)
    write_rfe_report(gene_pipeline, genes_after_k_best, write_line)
    write_final_coefficient_report(gene_pipeline, genes_after_rfe, write_line)
    write_reproducibility_json(gene_pipeline, genes_after_k_best)

    return report_path


# Save ANOVA F-scores and p-values for genes after variance filtering.
def write_anova_report(gene_pipeline, genes_after_variance, write_line):
    k_best_step = gene_pipeline.named_steps["k_best"]
    selector = getattr(k_best_step, "selector_", None)

    write_line("\n--- SelectKBest ANOVA ---")
    if selector is None:
        write_line("SelectKBest selector was not found.")
        return

    scores = getattr(selector, "scores_", None)
    p_values = getattr(selector, "pvalues_", None)
    write_line(f"SelectKBest saw {len(genes_after_variance)} features after variance filtering")

    if scores is None:
        write_line("SelectKBest scores are unavailable.")
        return

    anova_results = pd.DataFrame(
        {
            "GENE": genes_after_variance,
            "F_value": scores,
            "p_value": p_values if p_values is not None else np.nan,
        }
    ).sort_values("F_value", ascending=False)

    anova_path = OUTPUT_DIR / f"rfe_anova_details_trial.csv"
    anova_results.to_csv(anova_path, index=False)
    write_line(f"Saved ANOVA results to: {anova_path}")
    write_line("Top 10 ANOVA genes: " + ", ".join(anova_results["GENE"].head(10).tolist()))


# Save RFE rankings for inspection later.
def write_rfe_report(gene_pipeline, genes_after_k_best, write_line):
    rfe_step = gene_pipeline.named_steps["rfe"]
    fitted_rfe = getattr(rfe_step, "rfe_", None)

    write_line("\n--- RFE ---")
    if fitted_rfe is None:
        write_line("Fitted RFE object was not found.")
        return

    write_line("Underlying sklearn RFE parameters:")
    write_line(str(fitted_rfe))
    write_line(f"n_features_to_select: {fitted_rfe.n_features_to_select}")
    write_line(f"step: {fitted_rfe.step}")
    write_line(f"estimator within RFE: {fitted_rfe.estimator}")

    ranking = getattr(fitted_rfe, "ranking_", None)
    support = getattr(fitted_rfe, "support_", None)
    if ranking is None or support is None:
        write_line("RFE ranking or support attributes are unavailable.")
        return

    write_line(f"RFE ranking: min={ranking.min()}, max={ranking.max()}, selected={support.sum()}")
    rfe_results = pd.DataFrame(
        {
            "GENE": genes_after_k_best,
            "RFE_rank": ranking,
            "selected": support,
        }
    ).sort_values("RFE_rank", ascending=True)

    rfe_path = OUTPUT_DIR / f"rfe_ranking_trial.csv"
    rfe_results.to_csv(rfe_path, index=False)
    selected_genes = rfe_results.loc[rfe_results["RFE_rank"] == 1, "GENE"].head(10).tolist()
    write_line(f"Wrote RFE ranking and selection mask to: {rfe_path}")
    write_line("Top 10 RFE-selected genes: " + ", ".join(selected_genes))


# Save final logistic-regression coefficients for the RFE-selected genes.
def write_final_coefficient_report(gene_pipeline, genes_after_rfe, write_line):
    classifier = gene_pipeline.named_steps["classifier"]
    write_line("\n--- Final classifier ---")
    write_line("Classifier params: " + json.dumps(classifier.get_params(), default=str))
    write_line(f"After variance filter -> KBest -> RFE: {len(genes_after_rfe)} features selected")

    coefficients = classifier.coef_.flatten()
    if len(coefficients) != len(genes_after_rfe):
        write_line("Mismatch between coefficients and selected genes.")
        write_line(f"coefficient length {len(coefficients)}, selected gene length {len(genes_after_rfe)}")
        write_line(
            f"coefficient min {coefficients.min()}, max {coefficients.max()}, "
            f"mean abs {np.mean(np.abs(coefficients))}"
        )
        return

    coefficient_results = pd.DataFrame(
        {
            "GENE": list(genes_after_rfe),
            "COEFFICIENT": coefficients,
        }
    )
    coefficient_results["ABS_COEF"] = coefficient_results["COEFFICIENT"].abs()
    coefficient_results = coefficient_results.sort_values("ABS_COEF", ascending=False)

    coefficient_path = OUTPUT_DIR / f"rfe_final_coeffs_trial.csv"
    coefficient_results.to_csv(coefficient_path, index=False)
    write_line(f"Saved final coefficients for selected genes to: {coefficient_path}")
    write_line("Top 10 genes by |coefficient|: " + ", ".join(coefficient_results["GENE"].head(10).tolist()))


# Save model and feature-selection settings needed to reproduce the run.
def write_reproducibility_json(gene_pipeline, genes_after_k_best):
    rfe_step = gene_pipeline.named_steps["rfe"]
    fitted_rfe = getattr(rfe_step, "rfe_", None)
    classifier = gene_pipeline.named_steps["classifier"]
    k_best_step = gene_pipeline.named_steps["k_best"]
    variance_filter = gene_pipeline.named_steps["variance_filter"]

    reproducibility_data = {
        "RFE": {
            "n_features_to_select": int(fitted_rfe.n_features_to_select) if fitted_rfe is not None else None,
            "step": int(fitted_rfe.step) if fitted_rfe is not None else None,
            "estimator": str(fitted_rfe.estimator) if fitted_rfe is not None else str(rfe_step.estimator),
        },
        "LogisticRegressionCV_final_classifier": classifier.get_params(),
        "SelectKBest_ANOVA": {
            "k_requested": k_best_step.k,
            "k_effective": len(genes_after_k_best),
        },
        "VarianceThreshold": {"threshold": variance_filter.threshold},
    }

    output_path = OUTPUT_DIR / f"rfe_repro_snippet_trial.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(reproducibility_data, output_file, indent=2, default=str)


# Rebuild final scaled train/test dataframes using the fitted gene-selection pipeline.
def make_final_scaled_dataframes(gene_pipeline, train_features, test_features):
    gene_names = train_features.columns.to_numpy()
    genes_after_variance, _, genes_after_rfe = get_selected_gene_sets(gene_pipeline, gene_names)

    if len(genes_after_variance) == 0:
        raise RuntimeError("VarianceThreshold kept zero genes. Check the threshold.")
    if len(genes_after_rfe) == 0:
        raise RuntimeError("RFE selected zero genes. Check RFE settings.")

    # Apply the fitted pipeline's selected gene names to the original dataframes.
    train_after_variance = train_features.loc[:, genes_after_variance].values.astype(float)
    test_after_variance = test_features.loc[:, genes_after_variance].values.astype(float)

    scaler = gene_pipeline.named_steps["scaler"]
    scaled_train = scaler.transform(train_after_variance)
    scaled_test = scaler.transform(test_after_variance)

    scaled_train_after_variance = pd.DataFrame(
        scaled_train,
        index=train_features.index,
        columns=genes_after_variance,
    )
    scaled_test_after_variance = pd.DataFrame(
        scaled_test,
        index=test_features.index,
        columns=genes_after_variance,
    )

    final_scaled_train = scaled_train_after_variance.loc[:, genes_after_rfe]
    final_scaled_test = scaled_test_after_variance.loc[:, genes_after_rfe]

    print(f"\nFinal scaled train shape: {final_scaled_train.shape}")
    print(f"Final scaled test shape: {final_scaled_test.shape}")
    return final_scaled_train, final_scaled_test, genes_after_rfe


# Export selected genes ordered by absolute classifier coefficient size.
def save_final_gene_coefficients(classifier, selected_genes):
    coefficients = pd.Series(classifier.coef_[0], index=selected_genes)
    sorted_genes = coefficients.abs().sort_values(ascending=False).index
    coefficient_frame = pd.DataFrame(
        {
            "GENE": sorted_genes,
            "COEFFICIENT": coefficients.loc[sorted_genes].values,
        }
    )
    output_path = OUTPUT_DIR / f"paper.xlsx"
    coefficient_frame.to_excel(output_path, index=False)
    print(f"\nSaved gene coefficients to: {output_path}")
    return coefficient_frame, output_path


# Save a one-row CSV containing only PCA model performance metrics.
def save_pca_summary_metrics(pca_metrics, pca_auc_scores, pca_accuracy_scores, test_labels):
    summary = pd.DataFrame(
        {
            "method": ["PCA_model"],
            "accuracy": [pca_metrics.get("accuracy", np.nan)],
            "balanced_accuracy": [pca_metrics.get("balanced_accuracy", np.nan)],
            "precision": [pca_metrics.get("precision", np.nan)],
            "recall": [pca_metrics.get("recall", np.nan)],
            "f1": [pca_metrics.get("f1", np.nan)],
            "roc_auc": [pca_metrics.get("roc_auc", np.nan)],
            "avg_precision": [pca_metrics.get("avg_precision", np.nan)],
            "cohen_kappa": [pca_metrics.get("cohen_kappa", np.nan)],
            "cv_auc_mean": [np.mean(pca_auc_scores)],
            "cv_auc_std": [np.std(pca_auc_scores)],
            "cv_acc_mean": [np.mean(pca_accuracy_scores)],
            "cv_acc_std": [np.std(pca_accuracy_scores)],
            "test_samples": [len(test_labels)],
        }
    )
    output_path = OUTPUT_DIR / f"pca_analysis_scores_summary_.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nSaved PCA-only summary metrics to {output_path}")
    return summary, output_path


# Save a text report containing PCA test metrics and PCA cross-validation scores.
def save_pca_detailed_metrics(pca_metrics, pca_auc_scores, pca_accuracy_scores):
    output_path = OUTPUT_DIR / f"pca_analysis_scores_details_.txt"
    with open(output_path, "w", encoding="utf-8") as detail_file:
        detail_file.write("PCA model metrics\n")
        detail_file.write("-----------------\n")
        for metric_name, metric_value in pca_metrics.items():
            detail_file.write(f"{metric_name}:\n{metric_value}\n\n")
        detail_file.write("\nPCA CV ROC AUC scores:\n")
        detail_file.write(np.array2string(pca_auc_scores, precision=3) + "\n")
        detail_file.write("\nPCA CV accuracy scores:\n")
        detail_file.write(np.array2string(pca_accuracy_scores, precision=3) + "\n")
    print(f"Saved PCA-only detailed metrics to {output_path}")
    return output_path


# Extract selected elastic-net hyperparameters from a fitted LogisticRegressionCV model.
def extract_logistic_cv_info(classifier):
    selected_c = None
    selected_l1_ratio = None

    c_values = getattr(classifier, "C_", None)
    if c_values is not None:
        c_array = np.asarray(c_values).ravel()
        if c_array.size > 0:
            selected_c = float(c_array[0])

    l1_ratio_values = getattr(classifier, "l1_ratio_", None)
    if l1_ratio_values is not None:
        l1_ratio_array = np.asarray(l1_ratio_values).ravel()
        if l1_ratio_array.size > 0:
            selected_l1_ratio = float(l1_ratio_array[0])

    return {
        "C_selected": selected_c,
        "l1_ratio_selected": selected_l1_ratio,
        "params": classifier.get_params(),
    }


# Evaluate one manually fitted outer fold using that fold's selected genes.
def evaluate_outer_fold_pipeline(pipeline, validation_features, validation_labels, genes_after_variance, genes_after_rfe):
    if len(genes_after_rfe) == 0:
        return np.nan, np.nan

    classifier = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["scaler"]
    validation_after_variance = validation_features.loc[:, genes_after_variance].values.astype(float)
    scaled_validation = scaler.transform(validation_after_variance)
    validation_frame = pd.DataFrame(
        scaled_validation,
        index=validation_features.index,
        columns=genes_after_variance,
    )
    final_validation_features = validation_frame.loc[:, genes_after_rfe]
    probabilities = classifier.predict_proba(final_validation_features.values)[:, 1]
    predictions = classifier.predict(final_validation_features.values)
    fold_auc = roc_auc_score(validation_labels, probabilities) if len(np.unique(validation_labels)) > 1 else np.nan
    fold_accuracy = accuracy_score(validation_labels, predictions)
    return fold_auc, fold_accuracy


def run_manual_outer_cv(gene_pipeline, train_features, train_labels):
    print("\nRunning manual outer-CV loop to capture per-fold hyperparameters and coefficients...")
    # random_state=None
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=None)
    gene_names = train_features.columns.to_numpy()
    fold_reports = []
    coefficient_frames = []

    for fold_index, (fold_train_index, fold_validation_index) in enumerate(
        outer_cv.split(train_features.values, train_labels),
        start=1,
    ):
        print(f"\nFitting outer fold {fold_index}/{outer_cv.get_n_splits()}...")
        fold_train_features = pd.DataFrame(
            train_features.values[fold_train_index],
            index=train_features.index[fold_train_index],
            columns=gene_names,
        )
        fold_train_labels = pd.Series(
            train_labels.values[fold_train_index],
            index=train_features.index[fold_train_index],
        )
        fold_validation_features = pd.DataFrame(
            train_features.values[fold_validation_index],
            index=train_features.index[fold_validation_index],
            columns=gene_names,
        )
        fold_validation_labels = pd.Series(
            train_labels.values[fold_validation_index],
            index=train_features.index[fold_validation_index],
        )

        # Clone the pipeline so each fold is fit independently.
        fitted_pipeline = clone(gene_pipeline)
        fitted_pipeline.fit(fold_train_features.values, fold_train_labels)

        classifier = fitted_pipeline.named_steps["classifier"]
        genes_after_variance, _, genes_after_rfe = get_selected_gene_sets(fitted_pipeline, gene_names)
        coefficient_series = pd.Series(dtype=float)

        coefficients = classifier.coef_.flatten()
        if len(coefficients) == len(genes_after_rfe):
            coefficient_series = pd.Series(coefficients, index=genes_after_rfe)

        classifier_info = extract_logistic_cv_info(classifier)
        try:
            fold_auc, fold_accuracy = evaluate_outer_fold_pipeline(
                fitted_pipeline,
                fold_validation_features,
                fold_validation_labels,
                genes_after_variance,
                genes_after_rfe,
            )
        except Exception as error:
            print("Warning: validation evaluation failed:", error)
            fold_auc = np.nan
            fold_accuracy = np.nan

        fold_reports.append(
            {
                "fold": fold_index,
                "n_train": len(fold_train_index),
                "n_val": len(fold_validation_index),
                "n_selected_genes": len(genes_after_rfe),
                "auc_val": float(fold_auc) if not np.isnan(fold_auc) else None,
                "acc_val": float(fold_accuracy) if not np.isnan(fold_accuracy) else None,
                "C_selected": classifier_info.get("C_selected"),
                "l1_ratio_selected": classifier_info.get("l1_ratio_selected"),
            }
        )

        if not coefficient_series.empty:
            coefficient_frame = coefficient_series.rename(f"fold{fold_index}").to_frame()
            coefficient_frames.append(coefficient_frame)
            fold_coefficient_path = OUTPUT_DIR / f"elasticnet_fold{fold_index}_coefs_trial.csv"
            coefficient_frame.to_csv(fold_coefficient_path)

    fold_report_frame = pd.DataFrame(fold_reports)
    fold_report_path = OUTPUT_DIR / f"elasticnet_outer_cv_fold_reports_trial.csv"
    fold_report_frame.to_csv(fold_report_path, index=False)
    print(f"Wrote outer-fold CV report to: {fold_report_path}")

    merged_coefficients = save_outer_cv_coefficient_summary(coefficient_frames)
    save_outer_cv_reproducibility_json(gene_pipeline, outer_cv)
    return fold_report_frame, merged_coefficients


# Combine fold coefficient files and summarize how often each gene is selected.
def save_outer_cv_coefficient_summary(coefficient_frames):
    if not coefficient_frames:
        print("No coefficients collected across folds.")
        return pd.DataFrame()

    merged_coefficients = pd.concat(coefficient_frames, axis=1).fillna(0.0)
    coefficient_matrix_path = OUTPUT_DIR / f"elasticnet_coef_matrix_outerfolds_trial.csv"
    merged_coefficients.to_csv(coefficient_matrix_path)
    print("Saved coefficient matrix.")

    # Nonzero coefficient in a fold means the gene contributed to that fold's final classifier.
    selection_frequency = (merged_coefficients != 0).sum(axis=1) / merged_coefficients.shape[1]
    mean_coefficient_if_selected = merged_coefficients.replace(0.0, np.nan).mean(axis=1).fillna(0.0)
    selection_summary = pd.DataFrame(
        {
            "GENE": merged_coefficients.index,
            "selection_freq": selection_frequency.values,
            "mean_coef_if_selected": mean_coefficient_if_selected.values,
            "mean_abs_coef": merged_coefficients.abs().mean(axis=1).values,
        }
    ).sort_values("selection_freq", ascending=False)

    summary_path = OUTPUT_DIR / f"elasticnet_selection_summary_trial.csv"
    selection_summary.to_csv(summary_path, index=False)
    print("Saved selection frequency summary.")

    top_gene_count = min(50, selection_summary.shape[0])
    top_genes = selection_summary.head(top_gene_count).copy()
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top_genes["GENE"][::-1], top_genes["selection_freq"][::-1])
    axis.set_xlabel("Selection frequency (outer folds)")
    axis.set_title("Elastic-net: selection frequency of top genes (outer folds)")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"elasticnet_selection_freq_top{top_gene_count}_trial.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved selection frequency plot: {plot_path}")
    return merged_coefficients


# Save outer-CV and feature-selection settings for reproducibility.
def save_outer_cv_reproducibility_json(gene_pipeline, outer_cv):
    reproducibility_data = {
        "outer_cv_n_splits": outer_cv.get_n_splits(),
        "LogisticRegressionCV_params": gene_pipeline.named_steps["classifier"].get_params(),
        "RFE_params": gene_pipeline.named_steps["rfe"].get_params(),
        "SelectKBest_params": getattr(gene_pipeline.named_steps["k_best"], "__dict__", {}),
    }
    output_path = OUTPUT_DIR / f"elasticnet_repro_snippet_trial.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(reproducibility_data, output_file, indent=2, default=str)
    print("Saved reproducibility snippet.")


# Save ROC and precision-recall curves from class-1 predicted probabilities.
def plot_roc_and_precision_recall(labels, probabilities, filename_prefix):
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    roc_path = OUTPUT_DIR / f"{filename_prefix}_roc_trial.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    precision_recall_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"PR (AUC = {precision_recall_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    precision_recall_path = OUTPUT_DIR / f"{filename_prefix}_pr_trial.png"
    plt.savefig(precision_recall_path, dpi=300)
    plt.close()

    return roc_path, precision_recall_path


# Main pipeline
def main():
    configure_environment()

    expression_matrix = load_expression_matrix()
    nk_labels = load_nk_labels()
    labeled_expression, labeled_labels = align_expression_and_labels(expression_matrix, nk_labels)
    train_features, test_features, train_labels, test_labels = split_train_test(labeled_expression, labeled_labels)

    pca_pipeline = make_pca_pipeline()
    pca_auc_scores, pca_accuracy_scores = cross_validate_pipeline(
        pca_pipeline,
        train_features,
        train_labels,
        "PCA pipeline",
    )

    print("Fitting final PCA pipeline on full training set...")
    pca_pipeline.fit(train_features.values, train_labels)
    pca_predictions = pca_pipeline.predict(test_features.values)
    pca_probabilities = pca_pipeline.predict_proba(test_features.values)[:, 1]
    pca_metrics = evaluate_classifier(test_labels, pca_predictions, pca_probabilities)
    print_metrics("PCA-model evaluation on test set", pca_metrics)

    # Calculate the variance threshold from training data
    gene_variances = train_features.astype(float).var(axis=0, ddof=1)
    variance_threshold = np.percentile(gene_variances, VARIANCE_PERCENTILE)
    print(f"\nVariance threshold ({VARIANCE_PERCENTILE}th percentile) = {variance_threshold:.6g}")
    plot_variance_histograms(train_features, variance_threshold)

    # Fit the interpretable gene-selection pipeline after evaluating the PCA baseline.
    gene_pipeline = make_gene_pipeline(variance_threshold)
    gene_auc_scores, gene_accuracy_scores = cross_validate_pipeline(
        gene_pipeline,
        train_features,
        train_labels,
        "Gene-space pipeline",
    )

    print("Fitting final gene-space pipeline on full training set...")
    gene_pipeline.fit(train_features.values, train_labels)
    write_feature_selection_report(gene_pipeline, train_features)

    _, final_scaled_test, selected_genes = make_final_scaled_dataframes(
        gene_pipeline,
        train_features,
        test_features,
    )

    gene_classifier = gene_pipeline.named_steps["classifier"]
    gene_predictions = gene_classifier.predict(final_scaled_test.values)
    gene_probabilities = gene_classifier.predict_proba(final_scaled_test.values)[:, 1]
    gene_metrics = evaluate_classifier(test_labels, gene_predictions, gene_probabilities)

    final_gene_coefficients, _ = save_final_gene_coefficients(gene_classifier, selected_genes)

    #PCA metrics are saved in these final metric-output files.
    save_pca_summary_metrics(
        pca_metrics,
        pca_auc_scores,
        pca_accuracy_scores,
        test_labels,
    )
    save_pca_detailed_metrics(
        pca_metrics,
        pca_auc_scores,
        pca_accuracy_scores,
    )

    run_manual_outer_cv(gene_pipeline, train_features, train_labels)

    roc_path, precision_recall_path = plot_roc_and_precision_recall(
        test_labels,
        pca_probabilities,
        filename_prefix="pca_model",
    )
    print(f"Saved ROC: {roc_path}")
    print(f"Saved PR: {precision_recall_path}")
    print("\nAll done. Pipelines trained, reports exported, and PCA-only metrics saved.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
