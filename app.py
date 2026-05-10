"""
FaultSim Backend — Flask + TensorFlow
Real MNIST model with live fault injection, detection, and healing via REST API
"""

import os
import copy
import time
import threading
import numpy as np
import tensorflow as tf
import sklearn
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────

state = {
    "status":           "idle",
    "train_log":        [],
    "baseline_acc":     None,
    "baseline_loss":    None,
    "epochs_done":      0,
    "total_epochs":     5,
    "error":            None,
    "last_faulty_acc":  None,
    "last_faulty_loss": None,
    "last_healed_acc":  None,
    "last_healed_loss": None,
}

model              = None
base_weights       = None
faulty_model_global = None   # Stores the most recent injected model
detector           = None    # FaultDetector instance (built after training)
healer             = None    # Healer instance (built after training)
x_test             = None
y_test             = None

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────

def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ])
    m.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    return m


# ─────────────────────────────────────────────
# FAULT INJECTOR
# ─────────────────────────────────────────────

class FaultInjector:
    def __init__(self, seed=42):
        np.random.seed(seed)

    def inject(self, src_model, fault_type, severity=0.1, target_layers=None):
        corrupted = build_model()
        corrupted.set_weights(src_model.get_weights())

        layers  = corrupted.layers
        targets = target_layers if target_layers is not None else range(len(layers))

        affected = []
        for i in targets:
            if i >= len(layers):
                continue
            layer   = layers[i]
            weights = layer.get_weights()
            if not weights:
                continue

            if   fault_type == "weight_noise":       weights = self._inject_weight_noise(weights, severity)
            elif fault_type == "weight_zero":        weights = self._inject_weight_zero(weights, severity)
            elif fault_type == "neuron_failure":     weights = self._inject_neuron_failure(weights, severity)
            elif fault_type == "layer_degradation":  weights = self._inject_layer_degradation(weights, severity)
            elif fault_type == "random_swap":        weights = self._inject_random_swap(weights)
            else: continue

            layer.set_weights(weights)
            affected.append(i)

        return corrupted, affected

    def _inject_weight_noise(self, weights, severity):
        return [w + np.random.normal(0, severity, size=w.shape) for w in weights]

    def _inject_weight_zero(self, weights, severity):
        return [w * (np.random.rand(*w.shape) > severity) for w in weights]

    def _inject_neuron_failure(self, weights, severity):
        new_weights = []
        for w in weights:
            w = w.copy()
            if w.ndim == 2:
                n      = w.shape[1]
                failed = np.random.choice(n, int(severity * n), replace=False)
                w[:, failed] = 0
            elif w.ndim == 1:
                n      = len(w)
                failed = np.random.choice(n, int(severity * n), replace=False)
                w[failed] = 0
            new_weights.append(w)
        return new_weights

    def _inject_layer_degradation(self, weights, severity):
        return [w * (1 - severity) for w in weights]

    def _inject_random_swap(self, weights):
        new_weights = []
        for w in weights:
            flat = w.flatten().copy()
            np.random.shuffle(flat)
            new_weights.append(flat.reshape(w.shape))
        return new_weights


injector = FaultInjector()


# ─────────────────────────────────────────────
# FAULT DETECTOR
# ─────────────────────────────────────────────

class FaultDetector:
    """
    Detects faults by comparing per-layer mean activation statistics
    between the baseline model and a (potentially) corrupted model.
    """

    def __init__(self, baseline_model, baseline_data):
        self.baseline_stats = self._compute_stats(baseline_model, baseline_data)

    def _compute_stats(self, mdl, data):
        """Returns list of (layer_name, mean_activation | None) for each layer."""
        stats = []
        for layer in mdl.layers:
            try:
                if hasattr(layer, "output") and layer.output is not None:
                    sub = tf.keras.Model(inputs=mdl.inputs, outputs=layer.output)
                    act = sub(data, training=False).numpy()
                    stats.append((layer.name, float(np.mean(act))))
                else:
                    stats.append((layer.name, None))
            except Exception:
                stats.append((layer.name, None))
        return stats

    def detect(self, check_model, data, threshold=0.2):
        """
        Compare check_model's activations against baseline.
        Returns (is_faulty: bool, layer_diffs: list[dict]).
        """
        current_stats = self._compute_stats(check_model, data)
        layer_diffs   = []
        is_faulty     = False

        min_len = min(len(current_stats), len(self.baseline_stats))
        for i in range(min_len):
            b_name, b_val = self.baseline_stats[i]
            c_name, c_val = current_stats[i]

            if b_val is None or c_val is None:
                continue

            diff    = abs(c_val - b_val)
            flagged = diff > threshold
            if flagged:
                is_faulty = True

            layer_diffs.append({
                "layer_idx":     i,
                "layer_name":    b_name,
                "baseline_mean": round(b_val, 6),
                "current_mean":  round(c_val, 6),
                "diff":          round(diff,  6),
                "flagged":       flagged,
            })

        return is_faulty, layer_diffs


# ─────────────────────────────────────────────
# HEALER
# ─────────────────────────────────────────────

class Healer:
    """
    Repairs a corrupted model by blending its weights back toward
    the stored original (baseline) weights.
    """

    def __init__(self, original_weights):
        self.original_weights = [w.copy() for w in original_weights]

    def rollback(self, target_model, strength=0.5):
        """
        Blend: new_w = (1 - strength) * corrupted_w + strength * original_w
        strength=1.0 → full restore; strength=0.0 → no change.
        """
        new_weights = []
        for w, orig in zip(target_model.get_weights(), self.original_weights):
            repaired = (1.0 - strength) * w + strength * orig
            new_weights.append(repaired)
        target_model.set_weights(new_weights)
        return target_model

    def repair_layer(self, target_model, layer_idx):
        """Fully restore a single layer to its original weights."""
        weight_offset      = 0
        layer_weight_map   = {}
        for i, layer in enumerate(target_model.layers):
            lw = layer.get_weights()
            if lw:
                layer_weight_map[i] = list(range(weight_offset, weight_offset + len(lw)))
                weight_offset += len(lw)
            else:
                layer_weight_map[i] = []

        new_weights = list(target_model.get_weights())
        for idx in layer_weight_map.get(layer_idx, []):
            new_weights[idx] = self.original_weights[idx].copy()
        target_model.set_weights(new_weights)
        return target_model


# ─────────────────────────────────────────────
# TRAINING  (runs in background thread)
# ─────────────────────────────────────────────

class EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        state["epochs_done"] = epoch + 1
        state["train_log"].append({
            "epoch":    epoch + 1,
            "accuracy": round(float(logs.get("accuracy", 0)) * 100, 2),
            "loss":     round(float(logs.get("loss", 0)), 4),
        })


def _train_thread(epochs):
    global model, base_weights, x_test, y_test, detector, healer

    try:
        state["status"]      = "training"
        state["train_log"]   = []
        state["epochs_done"] = 0
        state["error"]       = None

        (x_train, y_train), (xt, yt) = tf.keras.datasets.mnist.load_data()
        x_train = x_train / 255.0
        x_test  = xt     / 255.0
        y_test  = yt

        m = build_model()
        m.fit(x_train, y_train, epochs=epochs, callbacks=[EpochCallback()], verbose=0)

        loss, acc = m.evaluate(x_test, y_test, verbose=0)

        model        = m
        base_weights = m.get_weights()

        # Initialise detector and healer with fresh baseline
        detector = FaultDetector(m, x_test[:300])
        healer   = Healer(base_weights)

        state["baseline_acc"]  = round(float(acc)  * 100, 4)
        state["baseline_loss"] = round(float(loss),        4)
        state["status"]        = "ready"

    except Exception as e:
        state["status"] = "error"
        state["error"]  = str(e)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify({
        "status":           state["status"],
        "epochs_done":      state["epochs_done"],
        "total_epochs":     state["total_epochs"],
        "train_log":        state["train_log"],
        "baseline_acc":     state["baseline_acc"],
        "baseline_loss":    state["baseline_loss"],
        "last_faulty_acc":  state["last_faulty_acc"],
        "last_faulty_loss": state["last_faulty_loss"],
        "last_healed_acc":  state["last_healed_acc"],
        "last_healed_loss": state["last_healed_loss"],
        "error":            state["error"],
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    if state["status"] == "training":
        return jsonify({"error": "Already training"}), 409

    body   = request.get_json(silent=True) or {}
    epochs = int(body.get("epochs", 5))
    epochs = max(1, min(20, epochs))
    state["total_epochs"] = epochs

    t = threading.Thread(target=_train_thread, args=(epochs,), daemon=True)
    t.start()

    return jsonify({"message": f"Training started ({epochs} epochs)"})


@app.route("/api/inject", methods=["POST"])
def api_inject():
    global faulty_model_global

    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    body = request.get_json()
    if not body:
        return jsonify({"error": "No JSON body"}), 400

    fault_type    = body.get("fault_type",    "neuron_failure")
    severity      = float(body.get("severity", 0.3))
    target_layers = body.get("target_layers", [1, 3])

    valid = {"weight_noise","weight_zero","neuron_failure","layer_degradation","random_swap"}
    if fault_type not in valid:
        return jsonify({"error": f"Unknown fault type: {fault_type}"}), 400

    severity = max(0.0, min(1.0, severity))
    state["status"] = "injecting"

    before_weights = [w.copy() for w in model.get_weights()]

    faulty, affected = injector.inject(model, fault_type, severity, target_layers)
    faulty.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    loss, acc = faulty.evaluate(x_test, y_test, verbose=0, batch_size=512)

    # Store for subsequent detect / heal calls
    faulty_model_global = faulty
    state["last_faulty_acc"]  = round(float(acc)  * 100, 4)
    state["last_faulty_loss"] = round(float(loss),        4)
    state["last_healed_acc"]  = None   # reset on new injection
    state["last_healed_loss"] = None

    after_weights = faulty.get_weights()
    weight_deltas = []
    for i, (bw, aw) in enumerate(zip(before_weights, after_weights)):
        diff = np.abs(aw - bw)
        if diff.sum() > 0:
            weight_deltas.append({
                "layer":       i,
                "mean_delta":  round(float(diff.mean()), 6),
                "max_delta":   round(float(diff.max()),  6),
                "pct_changed": round(float((diff > 0).mean() * 100), 2),
            })

    state["status"] = "ready"

    return jsonify({
        "fault_type":      fault_type,
        "severity":        severity,
        "target_layers":   target_layers,
        "affected_layers": affected,
        "faulty_acc":      round(float(acc)  * 100, 4),
        "faulty_loss":     round(float(loss),       4),
        "baseline_acc":    state["baseline_acc"],
        "baseline_loss":   state["baseline_loss"],
        "acc_drop":        round(state["baseline_acc"] - float(acc) * 100, 4),
        "loss_increase":   round(float(loss) - state["baseline_loss"],     4),
        "weight_deltas":   weight_deltas,
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Run FaultDetector on the last injected model (or the baseline model
    if no injection has happened yet) and return per-layer activation diffs.
    """
    if model is None or detector is None:
        return jsonify({"error": "Model not trained yet"}), 400

    body        = request.get_json(silent=True) or {}
    threshold   = float(body.get("threshold",   0.2))
    sample_size = int(body.get("sample_size",   300))
    threshold   = max(0.01, min(1.0, threshold))
    sample_size = max(50,   min(1000, sample_size))

    check_model = faulty_model_global if faulty_model_global is not None else model
    sample      = x_test[:sample_size]

    state["status"] = "detecting"
    is_faulty, layer_diffs = detector.detect(check_model, sample, threshold)
    state["status"] = "ready"

    return jsonify({
        "is_faulty":    is_faulty,
        "threshold":    threshold,
        "sample_size":  sample_size,
        "layer_diffs":  layer_diffs,
        "faulty_acc":   state["last_faulty_acc"],
        "baseline_acc": state["baseline_acc"],
    })


@app.route("/api/heal", methods=["POST"])
def api_heal():
    """
    Apply the Healer to the last injected faulty model.
    Supports full rollback (blend toward baseline) or single-layer repair.
    """
    global faulty_model_global

    if model is None or healer is None:
        return jsonify({"error": "Model not trained yet"}), 400
    if faulty_model_global is None:
        return jsonify({"error": "No faulty model to heal — inject a fault first"}), 400

    body       = request.get_json(silent=True) or {}
    strength   = float(body.get("strength",    0.5))
    mode       = body.get("mode",              "rollback")   # "rollback" | "layer"
    layer_idx  = body.get("layer_idx",         None)

    strength = max(0.0, min(1.0, strength))
    state["status"] = "healing"

    # Snapshot faulty accuracy before healing (in case it was already modified)
    pre_heal_loss, pre_heal_acc = faulty_model_global.evaluate(
        x_test, y_test, verbose=0, batch_size=512
    )

    if mode == "layer" and layer_idx is not None:
        healer.repair_layer(faulty_model_global, int(layer_idx))
    else:
        healer.rollback(faulty_model_global, strength)

    faulty_model_global.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    loss, acc = faulty_model_global.evaluate(x_test, y_test, verbose=0, batch_size=512)

    healed_acc  = round(float(acc)  * 100, 4)
    healed_loss = round(float(loss),       4)

    state["last_healed_acc"]  = healed_acc
    state["last_healed_loss"] = healed_loss
    state["status"]           = "ready"

    faulty_acc_before = round(float(pre_heal_acc) * 100, 4)
    acc_recovery      = round(healed_acc - faulty_acc_before, 4)

    return jsonify({
        "mode":           mode,
        "strength":       strength,
        "healed_acc":     healed_acc,
        "healed_loss":    healed_loss,
        "baseline_acc":   state["baseline_acc"],
        "baseline_loss":  state["baseline_loss"],
        "faulty_acc":     faulty_acc_before,
        "acc_recovery":   acc_recovery,
        "acc_remaining_gap": round(state["baseline_acc"] - healed_acc, 4),
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global model, faulty_model_global
    if model is None or base_weights is None:
        return jsonify({"error": "No baseline weights available"}), 400

    model.set_weights(base_weights)
    faulty_model_global           = None
    state["last_faulty_acc"]      = None
    state["last_faulty_loss"]     = None
    state["last_healed_acc"]      = None
    state["last_healed_loss"]     = None
    return jsonify({"message": "Model reset to baseline weights"})


@app.route("/api/sample_predictions", methods=["GET"])
def api_sample_predictions():
    if model is None:
        return jsonify({"error": "Model not trained"}), 400

    check = faulty_model_global if faulty_model_global is not None else model
    prob_model = tf.keras.Sequential([check, tf.keras.layers.Softmax()])
    preds  = prob_model(x_test[:10]).numpy().tolist()
    labels = y_test[:10].tolist()

    return jsonify({
        "predictions": [
            {
                "true_label": int(labels[i]),
                "pred_label": int(np.argmax(preds[i])),
                "confidence": round(float(max(preds[i])) * 100, 2),
                "correct":    int(labels[i]) == int(np.argmax(preds[i])),
                "probs":      [round(p, 4) for p in preds[i]],
            }
            for i in range(10)
        ]
    })


@app.route("/api/scores", methods=["POST"])
def api_scores():
    """
    Compute F1 (macro) and ROC-AUC (macro OvR) for the current model
    (faulty if injected, else baseline).  Also returns per-class F1,
    per-class AUC, and the full ROC curve points for all 10 classes.
    Accepts optional body: { "sample_size": 1000, "model": "baseline"|"faulty"|"healed" }
    """
    global faulty_model_global

    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    body        = request.get_json(silent=True) or {}
    sample_size = int(body.get("sample_size", 2000))
    model_choice = body.get("model", "current")   # "baseline" | "faulty" | "current"

    sample_size = max(200, min(10000, sample_size))

    # Choose which model to score
    if model_choice == "baseline":
        score_model = model
    elif model_choice == "faulty" and faulty_model_global is not None:
        score_model = faulty_model_global
    else:
        score_model = faulty_model_global if faulty_model_global is not None else model

    x_sample = x_test[:sample_size]
    y_true   = y_test[:sample_size]

    # Softmax probabilities
    prob_model = tf.keras.Sequential([score_model, tf.keras.layers.Softmax()])
    probs = prob_model(x_sample, training=False).numpy()   # (N, 10)
    y_pred = np.argmax(probs, axis=1)

    # ── F1 scores ──────────────────────────────────────────────────
    f1_macro  = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))
    f1_micro  = float(f1_score(y_true, y_pred, average="micro",  zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # ── ROC-AUC ────────────────────────────────────────────────────
    # One-hot encode true labels for multi-class AUC
    n_classes = 10
    y_true_oh = np.eye(n_classes)[y_true]

    try:
        auc_macro = float(roc_auc_score(y_true_oh, probs, average="macro",    multi_class="ovr"))
        auc_weighted = float(roc_auc_score(y_true_oh, probs, average="weighted", multi_class="ovr"))
    except Exception:
        auc_macro    = None
        auc_weighted = None

    # Per-class AUC and ROC curves
    per_class = []
    for c in range(n_classes):
        try:
            auc_c = float(roc_auc_score(y_true_oh[:, c], probs[:, c]))
            fpr, tpr, _ = roc_curve(y_true_oh[:, c], probs[:, c])
            # Downsample curve to ≤ 50 points for JSON size
            step = max(1, len(fpr) // 50)
            per_class.append({
                "class":   c,
                "f1":      round(f1_per_class[c], 4),
                "auc":     round(auc_c, 4),
                "fpr":     [round(float(v), 4) for v in fpr[::step]],
                "tpr":     [round(float(v), 4) for v in tpr[::step]],
                "support": int(np.sum(y_true == c)),
            })
        except Exception:
            per_class.append({
                "class": c, "f1": round(f1_per_class[c], 4),
                "auc": None, "fpr": [], "tpr": [],
                "support": int(np.sum(y_true == c)),
            })

    return jsonify({
        "model_choice":  model_choice,
        "sample_size":   sample_size,
        "f1_macro":      round(f1_macro,    4),
        "f1_micro":      round(f1_micro,    4),
        "f1_weighted":   round(f1_weighted, 4),
        "auc_macro":     round(auc_macro,   4) if auc_macro    is not None else None,
        "auc_weighted":  round(auc_weighted,4) if auc_weighted is not None else None,
        "per_class":     per_class,
        "baseline_acc":  state["baseline_acc"],
    })


@app.route("/")
def index():
    return "<p>FaultSim API running. Open <b>fault_injection_gui.html</b> in your browser.</p>"


if __name__ == "__main__":
    print("=" * 55)
    print("  FaultSim Backend  —  http://127.0.0.1:5000")
    print("  Open fault_injection_gui.html in your browser")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
