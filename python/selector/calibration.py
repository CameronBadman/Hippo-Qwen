from __future__ import annotations

from python.selector.dataset import AUXILIARY_LABELS


AuxiliarySamples = dict[str, list[tuple[float, bool]]]
AuxiliaryCounts = dict[str, dict[str, int]]


def default_auxiliary_thresholds(value: float = 0.5) -> dict[str, float]:
    return {label: value for label in AUXILIARY_LABELS}


def summarize_auxiliary(totals: AuxiliaryCounts) -> dict:
    per_label = {}
    correct = 0
    total = 0
    macro_f1 = 0.0
    labelled = 0
    for label in AUXILIARY_LABELS:
        counts = totals[label]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        tn = counts["tn"]
        label_total = tp + fp + fn + tn
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        correct += tp + tn
        total += label_total
        macro_f1 += f1
        labelled += 1
    return {
        "bit_accuracy": correct / max(1, total),
        "macro_f1": macro_f1 / max(1, labelled),
        "per_label": per_label,
    }


def tune_auxiliary_thresholds(samples: AuxiliarySamples) -> dict:
    thresholds = [index / 20.0 for index in range(1, 20)]
    per_label = {}
    totals = {}
    for label in AUXILIARY_LABELS:
        best_threshold = 0.5
        best_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        best_f1 = -1.0
        for threshold in thresholds:
            counts = counts_for_threshold(samples[label], threshold)
            precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
            recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
            f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
            if f1 > best_f1 or (f1 == best_f1 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_f1 = f1
                best_threshold = threshold
                best_counts = counts
        totals[label] = best_counts
        per_label[label] = {"threshold": best_threshold, "f1": best_f1}
    summary = summarize_auxiliary(totals)
    for label, details in per_label.items():
        summary["per_label"][label]["threshold"] = details["threshold"]
    summary["thresholds"] = {label: summary["per_label"][label]["threshold"] for label in AUXILIARY_LABELS}
    return summary


def counts_for_threshold(samples: list[tuple[float, bool]], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for score, expected in samples:
        predicted = score >= threshold
        if predicted and expected:
            counts["tp"] += 1
        elif predicted:
            counts["fp"] += 1
        elif expected:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts
