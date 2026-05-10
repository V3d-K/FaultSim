# FaultSim — Self-Healing Neural Network Fault Injection Framework

---

<img width="1910" height="978" alt="Screenshot 2026-05-10 160602" src="https://github.com/user-attachments/assets/e533ea9f-a3c7-4872-bd53-2c14fcf513f7" />

---

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

<img width="1909" height="976" alt="Screenshot 2026-05-10 160633" src="https://github.com/user-attachments/assets/b277dc31-2eab-45e8-9bc2-f469282c910c" />
<img width="1037" height="310" alt="Screenshot 2026-05-10 160655" src="https://github.com/user-attachments/assets/f7d5b0b1-7c6c-451e-aabb-68ae558463f3" />
<img width="1160" height="984" alt="Screenshot 2026-05-10 160749" src="https://github.com/user-attachments/assets/502bf386-6d68-473c-a454-a6de575a1445" />

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
