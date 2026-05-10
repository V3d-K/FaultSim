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

# Experimental Results

## Baseline Model Performance

| Metric | Value |
|---|---|
| Accuracy | 97.52% |
| Loss | 0.0806 |

---

## Key Findings

- Output layers were the most fault-sensitive
- Random weight swaps caused catastrophic collapse
- Weight noise produced severe instability
- Layer degradation showed highest robustness
- Non-trainable layers remained highly resilient

---

## Worst Fault Effects

| Fault Type | Worst Accuracy |
|---|---|
| Random Swap | ~8–10% |
| Weight Noise | ~13–18% |
| Neuron Failure | ~38–54% |
| Weight Zeroing | ~69–88% |

---

# Layer-Wise Sensitivity Ranking

| Rank | Layer | Sensitivity |
|---|---|---|
| 1 | Flatten Layer | Minimal |
| 2 | Dropout Layer | Minimal |
| 3 | Dense Hidden Layer | High |
| 4 | Dense Output Layer | Critical |

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

# Example Fault Injection

```python
injector = FaultInjector()

faulty_model = injector.inject(
    model,
    fault_type="neuron_failure",
    severity=0.3,
    target_layers=[1, 3]
)
```

---

# Example Workflow

```text
1. Train baseline model
2. Inject controlled faults
3. Monitor degradation
4. Detect abnormal behavior
5. Trigger healing mechanism
6. Re-evaluate performance
```

---

# Research Motivation

Traditional neural network robustness methods are mostly static and rely on retraining or predefined attack assumptions.

This project explores autonomous self-healing systems capable of:

- runtime degradation detection
- adaptive recovery
- lightweight repair
- resilience against unforeseen failures

The framework draws inspiration from:
- biological self-healing systems
- adaptive AI systems
- self-healing integrated circuits
- closed-loop neural control frameworks

---

# Future Improvements

Planned future work includes:

- transformer model support
- reinforcement-learning-based healing
- autoencoder-based reconstruction
- adaptive recovery policy selection
- distributed self-healing architectures
- adversarial defense integration
- hardware-level fault simulation

---

# References

1. Self-Healing Robust Neural Networks via Closed-Loop Control  
2. Self-Healing Neural Networks Against Adversarial Attacks  
3. AI-Based Approaches for Fault Detection and Recovery  
4. Repairing Deep Neural Networks: Fix Patterns and Challenges  
5. DeepPatch: Maintaining Deep Learning Model Programs  

---

# Author

Vedant Kanoje  
Senior Design Project  
VIT-AP University

---

# License

This project is intended for academic and research purposes.