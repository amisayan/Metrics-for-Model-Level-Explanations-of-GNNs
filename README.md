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
## Please run 4ShapesDemo.py which demonstrates the application of our metric and provides a complete workflow on the 4Shapes dataset. Other files will be updated post-acceptance of the paper and are placeholders for now.


---

## Installation

Clone the repository and install dependencies:

```bash
cd Metrics-for-Model-Level-Explanations-of-GNNs
pip install -r requirements.txt
python 4shapesDemo.py

