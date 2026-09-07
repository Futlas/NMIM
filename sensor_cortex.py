import numpy as np

class KigaiSensorCortex:
    """
    Сенсорная кора ядра KIGAI.
    Динамически нарезает текст на слоги с адаптивным хаос-окном,
    зависящим от стресса (кортизола), и транслирует их в гладкие
    пространственные волны для 64 каналов cl-sdk.
    """

    def __init__(self, channels_count: int = 64):
        self.channels_count = channels_count
   
        np.random.seed(42)
        self.W_in = np.random.uniform(0.1, 0.9, (1105, self.channels_count))
        self.previous_wave = np.zeros(self.channels_count, dtype=np.float64)
        self.sillywindow = 0

    def step_tokenize(self, text: str, cortisol: float, chaos_engine) -> np.ndarray: 
        """
        За один такт вырезает адаптивный слог из текста на основе уровня стресса
        и переданного живого объекта KigaiChaosEngine.
        """
        if not text:
            return np.zeros(self.channels_count, dtype=np.float64)
        
        if self.sillywindow >= len(text):
            self.sillywindow = 0
            self.previous_wave = np.zeros(self.channels_count, dtype=np.float64)
        
        base_window = 2
        if cortisol > 0.3:
            # Вызываем метод напрямую у ПЕРЕДАННОГО через аргументы живого объекта движка
            quantum_sample = chaos_engine.generate_quantum_noise()
            
            jit = int(np.round((quantum_sample - 0.5) * 4 * cortisol))
            window_size = int(np.clip(base_window + jit, 1, 4))
        else:
            window_size = base_window

        chunk = text[self.sillywindow : self.sillywindow + window_size]
        self.sillywindow += len(chunk)

        char_indices = np.array([min(ord(char), 1104) for char in chunk], dtype=np.int32)

        current_wave = np.zeros(self.channels_count, dtype=np.float64)
        for pos, idx in enumerate(char_indices):
            position_weight = 1.0 / (pos + 1)
            current_wave += self.W_in[idx] * position_weight

        final_wave = current_wave + self.previous_wave * 0.3
        final_wave = np.clip(final_wave, 0.0, 1.0)

        self.previous_wave = final_wave.copy()

        return final_wave
