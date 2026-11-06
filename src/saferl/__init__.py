"""Safe RL arms. Arm 1: vanilla DPO.

Importing this package pins transformers to the PyTorch backend: a TensorFlow install in the same
environment otherwise gets imported alongside torch and can deadlock (absl mutex) on macOS.
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
