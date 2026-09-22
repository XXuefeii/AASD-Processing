import numpy as np


def resolve_recursive_settings(settings, stride_s):
    result = dict(settings)
    if "recursive_time_constant_s" in result:
        if result["recursive_time_constant_s"] <= 0 or stride_s <= 0:
            raise ValueError("Recursive time constant and stride must be positive")
        result["recursive_n"] = result["recursive_time_constant_s"] / stride_s
        result["recursive_bias"] = -0.5 / result["recursive_n"]
    return result


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    f1 = []
    for c in (0, 1):
        denominator = cm[c].sum() + cm[:, c].sum()
        f1.append(float(2 * cm[c, c] / denominator) if denominator else 0.0)
    recalls = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None for c in (0, 1)]
    return {"accuracy": float(np.trace(cm) / len(labels)), "f1_right": f1[1],
            "f1_left": f1[0], "f1_macro": float(np.mean(f1)),
            "recall_left": recalls[0], "recall_right": recalls[1],
            "confusion_matrix": cm.tolist(), "n_windows": len(labels)}


def bounded_recursive(probabilities, n=5.0, bias=-0.1, initial_gain=0.5):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2 or not len(probabilities):
        raise ValueError("Expected nonempty windows x 2 probabilities")
    if n <= 0 or not 0 <= initial_gain <= 1:
        raise ValueError("Invalid recursive parameters")
    gain = np.full(2, initial_gain, dtype=np.float64)
    states, gains = [], []
    state = int(probabilities[0].argmax())
    for p in probabilities:
        gain = np.clip(gain + p / n + bias, 0, 1)
        if gain[0] != gain[1]:
            state = int(gain.argmax())
        states.append(state)
        gains.append(gain.copy())
    return np.asarray(states), np.asarray(gains)


def state_transitions(states, times):
    states, times = np.asarray(states), np.asarray(times)
    indexes = np.flatnonzero(states[1:] != states[:-1]) + 1
    return [(float(times[i]), int(states[i])) for i in indexes]


def match_events(truth, predicted, tolerance):
    n, m = len(truth), len(predicted)
    counts = np.zeros((n + 1, m + 1), dtype=int)
    costs = np.zeros((n + 1, m + 1), dtype=float)
    actions = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [(counts[i - 1, j], costs[i - 1, j], 1),
                       (counts[i, j - 1], costs[i, j - 1], 2)]
            delta = abs(truth[i - 1][0] - predicted[j - 1][0])
            if truth[i - 1][1] == predicted[j - 1][1] and delta <= tolerance:
                options.append((counts[i - 1, j - 1] + 1, costs[i - 1, j - 1] + delta, 3))
            best = max(options, key=lambda v: (v[0], -v[1], v[2] == 3))
            counts[i, j], costs[i, j], actions[i, j] = best
    pairs = []
    i, j = n, m
    while i and j:
        action = actions[i, j]
        if action == 3:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif action == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    signed = [predicted[j][0] - truth[i][0] for i, j in pairs]
    return {"n_true": n, "n_predicted": m, "n_matched": len(pairs),
            "false_positive": m - len(pairs), "false_negative": n - len(pairs),
            "sdl_s": float(np.mean(np.abs(signed))) if signed else None,
            "signed_delay_s": float(np.mean(signed)) if signed else None,
            "absolute_errors_s": list(map(float, np.abs(signed))),
            "signed_errors_s": signed,
            "matches": [{"true_time_s": truth[i][0], "predicted_time_s": predicted[j][0],
                         "direction": truth[i][1]} for i, j in pairs]}


def aggregate_events(trials):
    count = len(trials)
    true_count = sum(v["n_true"] for v in trials)
    fp = sum(v["false_positive"] for v in trials)
    fn = sum(v["false_negative"] for v in trials)
    absolute = [e for v in trials for e in v["absolute_errors_s"]]
    signed = [e for v in trials for e in v["signed_errors_s"]]
    return {"n_trials": count, "n_true": true_count, "n_matched": len(absolute),
            "false_positive_events": fp, "false_negative_events": fn,
            "false_positive_per_trial": fp / count if count else None,
            "false_negative_rate": fn / true_count if true_count else None,
            "sdl_s": float(np.mean(absolute)) if absolute else None,
            "signed_delay_s": float(np.mean(signed)) if signed else None}
