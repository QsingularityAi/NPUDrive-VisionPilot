"""Base class for TIDL ONNX Runtime models on TI J784S4 NPU."""
import os
import time
from pathlib import Path
import onnxruntime as ort
import numpy as np

class BaseTIDLModel:
    def __init__(self, name: str, onnx_path: str, artifacts_dir: str, prefer_tidl: bool = True):
        self.name = name
        self.onnx_path = str(onnx_path)
        self.artifacts_dir = str(artifacts_dir)
        self.prefer_tidl = prefer_tidl
        
        self.session = None
        self.active_provider = None
        self.input_names = []
        self.output_names = []
        self.last_latency_ms = 0.0
        
        self._init_session()

    def _init_session(self):
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_path}")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        available_providers = ort.get_available_providers()
        use_tidl = self.prefer_tidl and ("TIDLExecutionProvider" in available_providers) and os.path.exists(self.artifacts_dir)

        if use_tidl:
            opts = {
                "artifacts_folder": self.artifacts_dir,
                "debug_level": 0
            }
            providers = ["TIDLExecutionProvider", "CPUExecutionProvider"]
            provider_options = [opts, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        t0 = time.time()
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=so,
            providers=providers,
            provider_options=provider_options
        )
        load_time = (time.time() - t0) * 1000
        self.active_provider = self.session.get_providers()[0]
        
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"[{self.name.upper()}] Initialized with provider '{self.active_provider}' in {load_time:.1f}ms")
        print(f"[{self.name.upper()}] Inputs: {self.input_names} | Outputs: {self.output_names}")

    def run(self, feed_dict: dict):
        t0 = time.time()
        outputs = self.session.run(None, feed_dict)
        self.last_latency_ms = (time.time() - t0) * 1000.0
        return outputs
