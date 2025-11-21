# Metrics for Model-Level Explanations of GNNs

This repository contains the implementation of the metrics proposed in our ICLR 2026 submission:

- **Coverage**
- **Greedy Gain Area (GGA)**
- **Overlap**

The code also includes:
- Computation of the Lipschitz constant of the classifier head,
- Derivation of the angular radius \( r^* \),
- Hoeffding concentration bounds for Coverage and GGA.

---



---

## Installation

Clone the repository and install dependencies:

```bash
cd Metrics-for-Model-Level-Explanations-of-GNNs
pip install -r requirements.txt
