import numpy as np

from chaos_engine import KigaiChaosEngine
from chemistry_pool import KigaiChemistryPool


class VirtualBioNeuron:
    """
    Гибкий кубит-подобный узел ядра KIGAI (v0.1-silico).
    Реализует Марковское одеяло предиктивного кодирования Фристона.
    Фича: Двойной незначительный увод параметров (mu и pi) под кортизолом.
    """
    def __init__(self):
        # Гауссовские параметры синаптического ожидания (Базовое состояние)
        self.mu = 0.5         # Центр тяжести информации (матожидание)
        self.precision = 3.5  # ПУТЬ 3: Подняли базовую точность с 2.0 до 3.5
        
    def process_signal(self, 
                       input_amplitude: float, 
                       chemistry: KigaiChemistryPool, 
                       chaos: KigaiChaosEngine) -> float:
        """
        Вычисляет выходную амплитуду вероятности [0.0, 1.0] с учетом
        гормонального профиля и каскадного Тейлор-шума.
        """
        cortisol_level = chemistry.cortisol
        
        # Контур "Двойного незначительного увода"
        if cortisol_level > 0.1:
            noise = chaos.generate_quantum_noise()
            shift = noise * cortisol_level * 0.05
            
            # Канал МЮ: смещаем координату информации
            self.mu += shift
            self.mu = np.clip(self.mu, 0.0, 1.0)
            
            # Канал ПИ (precision): уменьшаем скорость выгорания точности синапса
            self.precision -= shift * 5.0
            self.precision = max(0.1, self.precision)
            
        # Предиктивная фильтрация сигнала (Active Inference)
        weighted_signal = (input_amplitude - self.mu) * self.precision
        
        # Выход через плавную сигмоиду (Марковское одеяло узла)
        output_probability = 1.0 / (1.0 + np.exp(-weighted_signal))
        
        return output_probability
