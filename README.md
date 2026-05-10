# FaultSim — Neural Fault Injection Workbench

Real TensorFlow MNIST model + live fault injection via Flask API,
controlled from a browser-based GUI.

## Quick start

```bash
# 1. Install dependencies (inside a venv recommended)
pip install -r requirements.txt

# 2. Start the backend
python app.py

# 3. Open the GUI in your browser
open fault_injection_gui.html     # macOS
xdg-open fault_injection_gui.html # Linux
# or just double-click the file on Windows
```

The Flask server runs at **http://127.0.0.1:5000**.  
The HTML file talks to it directly — no build step needed.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/status` | Model status, epoch progress, baseline metrics |
| POST | `/api/train`  | Start training. Body: `{"epochs": 5}` |
| POST | `/api/inject` | Inject a fault. See body below |
| POST | `/api/reset`  | Restore model to baseline weights |
| GET  | `/api/sample_predictions` | Softmax predictions on first 10 test images |

### Inject body
```json
{
  "fault_type":    "neuron_failure",
  "severity":      0.3,
  "target_layers": [1, 3]
}
```

### Fault types
| Name | Effect |
|------|--------|
| `neuron_failure` | Zeros random neurons — simulates stuck-at-zero hardware |
| `weight_noise` | Adds Gaussian noise N(0, severity) to weights |
| `weight_zero` | Randomly masks weights to 0 with p=severity |
| `layer_degradation` | Scales all weights by (1 − severity) |
| `random_swap` | Shuffles weight values within each tensor |

---

## Project layout
```
app.py                   Flask backend (TF model + fault injection engine)
fault_injection_gui.html Frontend GUI
requirements.txt         Python dependencies
```
