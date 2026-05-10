# FaultSim — Self-Healing Neural Network Fault Injection Framework

FaultSim is a research-oriented neural network fault injection and self-healing framework designed to simulate, detect, analyze, and recover from faults in deep learning systems.

The project focuses on improving neural network robustness and reliability using controlled fault injection, runtime anomaly detection, and lightweight autonomous healing strategies.

---

# Features

- Neural network fault injection system
- Runtime fault detection and monitoring
- Self-healing recovery pipeline
- Layer-wise fault sensitivity analysis
- Severity-controlled degradation testing
- Interactive GUI dashboard
- Real-time metric visualization
- Baseline vs faulty model comparison
- Accuracy and loss degradation tracking
- Closed-loop self-healing architecture

---

# Implemented Fault Types

| Fault Type | Description |
|---|---|
| Weight Noise | Adds Gaussian perturbation to model weights |
| Weight Zeroing | Randomly zeros model parameters |
| Neuron Failure | Disables neurons by zeroing activations |
| Layer Degradation | Gradually attenuates layer outputs |
| Random Weight Swap | Randomizes learned weight organization |

These faults simulate:
- hardware instability
- internal parameter corruption
- adversarial degradation
- neuron failures
- structural model corruption

---

# Self-Healing Pipeline

```text
Input
   ↓
Baseline Model
   ↓
Fault Injection
   ↓
Internal Monitoring
   ↓
Fault Detection
   ↓
Healing Strategy
   ↓
Performance Re-evaluation
```

The framework follows a closed-loop self-healing architecture inspired by biological recovery systems and adaptive AI control systems.

---

# Detection Mechanisms

The detection module monitors:

- activation statistics
- confidence degradation
- abnormal prediction behavior
- loss spikes
- layer-wise sensitivity changes

Detection techniques include:
- threshold-based anomaly analysis
- statistical deviation monitoring
- runtime degradation analysis

---

# Healing Methods

Current recovery strategies include:

| Method | Description |
|---|---|
| Weight Rollback | Restores weights toward healthy checkpoints |
| Recovery Blending | Partial rollback using adaptive blending |
| Local Fine-Tuning | Retrains only corrupted layers |
| Structural Stabilization | Attempts to preserve learned representations |

The healing system is designed to recover performance without full retraining.

---


# Tech Stack

## Backend
- Python
- TensorFlow
- Keras
- NumPy

## Frontend
- HTML5
- CSS3
- JavaScript

## Deep Learning
- CNN Models
- MLP Architectures
- Fault Injection Modules
- Statistical Monitoring Systems

---

# GUI Features

The FaultSim GUI includes:

- fault selection controls
- severity sliders
- architecture visualization
- healing controls
- monitoring dashboard
- performance analytics
- training progress tracking

---

# Project Structure

```text
FaultSim/
│
├── app.py
├── fault_inject.py
├── fault_injection_gui.html
├── models/
├── datasets/
├── notebooks/
├── results/
├── screenshots/
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/V3d-K/FaultSim.git
cd FaultSim
```
---

## Install Dependencies

```bash
pip install tensorflow numpy flask
```

---

# Running the Project

## Start Backend

```bash
python app.py
```

## Open GUI

Open:

```text
fault_injection_gui.html
```

in your browser.

---