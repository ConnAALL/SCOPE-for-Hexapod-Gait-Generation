import os
import json

import numpy as np
from scipy.fftpack import dct

# load config (now with HIDDEN_STATE)
config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.txt'))
with open(config_file, 'r') as f: config = json.load(f)

MODEL_TYPE     = config['MODEL_TYPE']
HIDDEN_STATE   = config.get('HIDDEN_STATE', 49)

class SCOPEController:
    def __init__(self,
                 chromosome: list,
                 sensor_input_sample: np.ndarray):
        # parse weights & biases as before
        self.forward = self.ssgaForward if MODEL_TYPE == 'ssga' else self.scopeForward
        self.update_weights(chromosome)

        self._prev_inputs = [np.zeros_like(sensor_input_sample) for _ in range(HIDDEN_STATE)]
    
    def construct_input(self, sensor_input: np.ndarray) -> np.ndarray:
        # Update the queue: insert latest input at position 0
        if HIDDEN_STATE > 0:
            self._prev_inputs.pop()  # remove the oldest (last)
            self._prev_inputs.insert(0, sensor_input.copy())  # insert newest at front

        # Concatenate from newest to oldest
        full_input = np.concatenate([sensor_input] + self._prev_inputs, axis=1)

        return full_input
    
    def ssgaForward(self,
                  sensor_input: np.ndarray) -> np.ndarray:
        complete_input = self.construct_input(sensor_input)
        Z = complete_input.flatten()
        Y = self.chromosome * Z + self.bias
        return Y
    
    def scopeForward(self,
                     sensor_input: np.ndarray) -> np.ndarray:
        complete_input = self.construct_input(sensor_input)

        # Compute the DCT of the input
        compressed = dct(dct(complete_input.T, norm='ortho').T, norm='ortho')[:6, :9]

        Y = self.chromosome * compressed.flatten() + self.bias

        return Y.flatten()
    
    def update_weights(self,
                       new_weights: list) -> None:
        """
        Update the weights of the SSVD controller.
        """
        self.chromosome = np.array(new_weights[:54])
        self.bias       = np.array(new_weights[54:])
