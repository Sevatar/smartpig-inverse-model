# SmartPIG inverse model

The untrained visual-transformation CNN (VT-CNN) used by the SmartPIG
inverse-model benchmark. The model maps a window of five neighboring sensor
channels and 61 axial samples to a 61 × 65 output patch.

This repository contains the architecture only. There are no trained weights,
training or field datasets, calibration constants, or cluster scripts.
The model starts with fresh Keras initializers. Its untrained outputs are not
calibrated depth estimates and should not be used to interpret inspections.

## Quick start

Use Python 3.10–3.12:

```sh
python -m pip install -e .
```

```python
import tensorflow as tf
from smartpig_inverse import build_vt_cnn

model = build_vt_cnn()
x = tf.zeros((1, 5, 61, 1), dtype=tf.float32)
y = model(x, training=False)
print(y.shape)  # (1, 61, 65)
```

The input axes are batch, neighboring sensors, axial samples, and a singleton
feature channel. Output axes are batch, axial samples, and circumferential bins.
Input preparation and assembly of overlapping output patches are outside the
scope of this repository.

Run the synthetic smoke test with:

```sh
python -m unittest discover -s tests -v
```
