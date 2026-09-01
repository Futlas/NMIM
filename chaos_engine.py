import time

import numpy as np


class KigaiChaosEngine:
    """
    Движок квантового детерминированного хаоса ядра KIGAI (v0.1-silico).
    Генерирует энтропию на базе наносекундного джиттера системного таймера железа.
    Реализует каскадную тригонометрическую матрёшку Тейлора с плавающим round().
    """
    def __init__(self):
        
        self._last_state = 0.5
    def generate_quantum_noise(self) -> float :
        """
        Вычисляет шаг хаотической волны. 
        """

        ns = time.time_ns()

        inner_doll = np.sin(self._last_state +np.sin(ns*0.000001))
        outer_doll = np.sin(inner_doll + np.cos(ns * 0.000007))

        current_state = (outer_doll +1.0) / 2.0

        accuracy = (ns % 6)+1

        collapsed_state = round(current_state,accuracy)

        self._last_state = collapsed_state

        return collapsed_state
        