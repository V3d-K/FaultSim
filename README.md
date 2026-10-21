# FaultSim — Neural Network Fault Injection & Self-Healing Workbench

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)

**FaultSim** is an interactive workbench for studying how neural networks break and how they can be repaired. Train a real MNIST classifier, corrupt it with controlled faults, detect the damage from internal activation statistics, heal it, and watch accuracy, loss, F1 and ROC-AUC change at every step, all from a browser dashboard.

<img width="1910" height="978" alt="FaultSim dashboard" src="https://github.com/user-attachments/assets/e533ea9f-a3c7-4872-bd53-2c14fcf513f7" />

---

## Table of Contents

- [Why FaultSim?](#why-faultsim)
- [Features](#features)
- [How It Works](#how-it-works)
- [Fault Types](#fault-types)
- [Detection](#detection)
- [Healing](#healing)
- [Quick Start](#quick-start)
- [Using the Dashboard](#using-the-dashboard)
- [REST API](#rest-api)
- [Project Structure](#project-structure)
- [Limitations & Roadmap](#limitations--roadmap)
- [License](#license)

---

## Why FaultSim?

Deployed neural networks face bit flips, degraded hardware, corrupted checkpoints and adversarial tampering. FaultSim gives you a small, hackable environment to:

- **Quantify** how much each fault type and severity hurts a model
- **Probe** which layers are most sensitive
- **Test** detection and recovery ideas in a closed loop, with immediate visual feedback
- **Teach** fault tolerance and model robustness concepts interactively

---

## Features

- **Live training** of a baseline MNIST model (1–20 epochs) with real-time progress
- **Five fault injectors** with adjustable severity and per-layer targeting
- **Activation-based fault detector** with configurable threshold and sample size
- **Two healing modes:** blended rollback and single-layer repair
- **Rich metrics:** accuracy, loss, macro/micro/weighted F1, macro/weighted ROC-AUC, per-class F1 and ROC curves
- **Weight delta inspection:** mean delta, max delta and percent of weights changed per layer
- **Run history** charts comparing baseline vs. faulty vs. healed across experiments
- **Sample predictions** viewer showing true label, predicted label and confidence
- **API console** in the GUI logging every backend call

---

## How It Works

```text
        ┌────────────┐
        │   Train    │  MNIST, Dense(128) → Dropout → Dense(10)
        └─────┬──────┘
              ▼
        ┌────────────┐
        │  Baseline  │  weights + per-layer activation stats saved
        └─────┬──────┘
              ▼
        ┌────────────┐
        │   Inject   │  corrupt a copy of the model
        └─────┬──────┘
              ▼
        ┌────────────┐
        │   Detect   │  compare activation means vs. baseline
        └─────┬──────┘
              ▼
        ┌────────────┐
        │    Heal    │  rollback blend or layer repair
        └─────┬──────┘
              ▼
        ┌────────────┐
        │ Re-evaluate│  accuracy, loss, F1, ROC-AUC
        └────────────┘
```

The Flask backend (`app.py`) owns the model and exposes a REST API. The single-file GUI (`FaultSim_gui.html`) talks to it over `http://127.0.0.1:5000` and renders everything with Chart.js.

**Default model**

| Layer index | Layer | Notes |
|---|---|---|
| 0 | `Flatten` | 28×28 → 784, no weights |
| 1 | `Dense(128, relu)` | |
| 2 | `Dropout(0.2)` | no weights |
| 3 | `Dense(10)` | logits output |

Faults are applied only to layers that have weights (indices 1 and 3 by default).

---

## Fault Types

| Fault | ID | What it does |
|---|---|---|
| Weight Noise | `weight_noise` | Adds Gaussian noise with σ = severity to weights and biases |
| Weight Zeroing | `weight_zero` | Zeros each parameter with probability = severity |
| Neuron Failure | `neuron_failure` | Zeros a severity-fraction of a layer's output neurons (weight columns and biases) |
| Layer Degradation | `layer_degradation` | Scales weights by `(1 − severity)` |
| Random Weight Swap | `random_swap` | Shuffles weights within each tensor, keeping the value distribution but destroying learned structure |

Severity is a value in `[0, 1]`. Faults are injected into a **copy** of the model, so the baseline stays intact.

These approximate real-world failure modes such as hardware instability, memory corruption, neuron death and structural damage.

---

## Detection

`FaultDetector` records the **mean activation of every layer** on a sample of test data from the healthy model. To check a model, it recomputes those means and flags any layer where the absolute difference exceeds a threshold.

- Configurable `threshold` (0.01–1.0, default 0.2) and `sample_size` (50–1000, default 300)
- Returns per-layer baseline mean, current mean, difference and a flagged/not-flagged status
- Lightweight: no labels required, only forward passes

---

## Healing

`Healer` repairs a corrupted model using the stored baseline weights.

| Mode | Description |
|---|---|
| **Rollback blend** | `new = (1 − s) · corrupted + s · baseline` across all layers. `s = 1.0` is a full restore, `s = 0` is a no-op |
| **Layer repair** | Fully restores a single chosen layer, useful for targeted recovery after detection pinpoints the damaged layer |

The response reports accuracy before and after healing, the recovery gained and the remaining gap to baseline.

---

## Quick Start

### Requirements

- Python 3.9+
- A modern web browser (the GUI loads Chart.js and fonts from a CDN, so an internet connection is needed on first load)

### Install

```bash
git clone https://github.com/V3d-K/FaultSim.git
cd FaultSim

# optional but recommended
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run

```bash
python app.py
```

The backend starts at `http://127.0.0.1:5000`. Then open **`FaultSim_gui.html`** directly in your browser (double-click or drag it into a tab).

> MNIST is downloaded automatically by Keras on first training run.

---

## Using the Dashboard

1. **Train** the baseline model. Choose the number of epochs and wait for it to finish.
2. **Inject** a fault: pick a type, set severity, choose target layers.
3. **Detect** the fault: tune the threshold and see which layers get flagged.
4. **Heal** it: choose rollback strength or repair a specific layer.
5. **Compare** results across the tabs below.
6. **Reset** to baseline weights and try another experiment.

| Tab | What you see |
|---|---|
| Run History | Accuracy and loss across all runs (baseline / faulty / healed) |
| Detection | Per-layer activation diffs with flagged layers highlighted |
| Heal Results | Faulty vs. healed vs. baseline accuracy |
| Weight Deltas | How much each layer's weights changed |
| F1 & ROC-AUC | Macro/micro/weighted F1, AUC, per-class ROC curves and F1 bars |
| Predictions | Ten sample test predictions with confidence |

<img width="1909" height="976" alt="FaultSim results view" src="https://github.com/user-attachments/assets/b277dc31-2eab-45e8-9bc2-f469282c910c" />
<img width="1037" height="310" alt="FaultSim detection view" src="https://github.com/user-attachments/assets/f7d5b0b1-7c6c-451e-aabb-68ae558463f3" />
<img width="1160" height="984" alt="FaultSim metrics view" src="https://github.com/user-attachments/assets/502bf386-6d68-473c-a454-a6de575a1445" />

---

## REST API

You can drive FaultSim without the GUI. All endpoints are served from `http://127.0.0.1:5000`.

| Method | Endpoint | Purpose | Key body fields |
|---|---|---|---|
| `GET` | `/api/status` | Training status, baseline and latest metrics | n/a |
| `POST` | `/api/train` | Start background training | `epochs` (1–20) |
| `POST` | `/api/inject` | Inject a fault, return metrics and weight deltas | `fault_type`, `severity`, `target_layers` |
| `POST` | `/api/detect` | Run activation-based detection | `threshold`, `sample_size` |
| `POST` | `/api/heal` | Heal the most recent faulty model | `mode` (`rollback` / `layer`), `strength`, `layer_idx` |
| `POST` | `/api/reset` | Restore baseline weights, clear fault state | n/a |
| `GET` | `/api/sample_predictions` | Predictions on 10 test images | n/a |
| `POST` | `/api/scores` | F1 and ROC-AUC (overall and per class) | `sample_size`, `model` (`baseline` / `faulty` / `current`) |

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/train  -H "Content-Type: application/json" -d '{"epochs": 5}'
curl -X POST http://127.0.0.1:5000/api/inject -H "Content-Type: application/json" \
     -d '{"fault_type": "weight_noise", "severity": 0.2, "target_layers": [1, 3]}'
curl -X POST http://127.0.0.1:5000/api/heal   -H "Content-Type: application/json" \
     -d '{"mode": "rollback", "strength": 0.8}'
```

---

## Project Structure

```text
FaultSim/
├── app.py              # Flask backend: model, FaultInjector, FaultDetector, Healer, REST API
├── FaultSim_gui.html   # Single-file dashboard (HTML/CSS/JS + Chart.js)
├── requirements.txt    # Python dependencies
├── LICENSE.txt         # MIT license
└── README.md
```

---

## Limitations & Roadmap

FaultSim is a research and teaching tool, not a production reliability system. Current caveats:

- **Healing uses a stored copy of the original weights**, so it demonstrates recovery dynamics rather than blind repair. Healing without a reference checkpoint is an open direction.
- **Detection is based on mean activations only**, which can miss faults that leave layer means unchanged (for example a random weight swap on some layers).
- **One model and dataset:** a small dense network on MNIST.
- **Single global session:** the backend keeps one model in memory and is meant for local, single-user use.

Ideas for future work:

- [ ] Local fine-tuning of only the corrupted layers
- [ ] Richer detectors (confidence drift, loss spikes, activation variance, KL divergence)
- [ ] Layer-wise sensitivity sweeps across severities
- [ ] Support for CNNs and other datasets
- [ ] Reference-free healing (pruning, redundancy, weight-statistics repair)
- [ ] Exportable experiment reports

Contributions, issues and ideas are welcome.

---
