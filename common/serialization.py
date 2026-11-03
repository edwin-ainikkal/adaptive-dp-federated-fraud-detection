"""
common/serialization.py
------------------------
Turns a dict of NumPy arrays (a model's weights, or an update) into a
JSON-safe base64 string so it can travel inside a normal Flask/requests
HTTP JSON payload between the two desktops, and back again.
"""

import base64
import pickle


def encode_state(state_np: dict) -> str:
    return base64.b64encode(pickle.dumps(state_np)).decode("utf-8")


def decode_state(payload: str) -> dict:
    return pickle.loads(base64.b64decode(payload.encode("utf-8")))
