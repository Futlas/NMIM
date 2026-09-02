import numpy as np

from chaos_engine import KigaiChaosEngine
from chemistry_pool import KigaiChemistryPool


class KigaiPopulationLayer:
    """
    Универсальная однородная кора нейронов НМИМ (v0.2-silico).
    """
    def __init__(self):
        # Размер и входные каналы коры
        self.size = 64
        self.input_dim = 64
        
        # Координаты и матрица расстояний между нейронами
        self.coords = np.random.uniform(0.1, 0.9, (self.size, 2))
        self.D = np.zeros((self.size, self.size), dtype=np.float64)
        
        # Синаптические матрицы весов
        self.W_in = np.random.uniform(0.1, 0.3, (self.size, self.input_dim))
        self.W_rec = np.random.uniform(0.05, 0.15, (self.size, self.size))

        np.fill_diagonal(self.W_rec,0.0)

        self.precision = np.ones(self.size,dtype=np.float64)*2.0
        self.last_state= np.zeros(self.size, dtype=np.float64)

        self.learning_rate = 0.01

        self.last_input = np.zeros(self.input_dim, dtype=np.float64)
        self.last_delta_output = np.zeros(self.size, dtype=np.float64)


    def update_spatial_topology(self,repulsion_base: float = 0.02):

        diff = self.coords[:,None,:] - self.coords[None,:,:]

        self.D = np.linalg.norm(diff,axis=-1)
        np.fill_diagonal(self.D,1.0)

        attraction = self.W_rec[:,:,None] * diff

        repulsion = (repulsion_base / (self.D **2 +1e-4))[:,:,None] * diff

        np.fill_diagonal(attraction[:, :, 0], 0.0)
        np.fill_diagonal(repulsion[:, :, 0], 0.0)

        spatial_shift = np.mean(attraction - repulsion, axis=1) * 0.05
        self.coords = np.clip(self.coords + spatial_shift, 0.0, 1.0)

    def process_population_signal(self, 
                                  raw_vector: np.ndarray, 
                                  chemistry: KigaiChemistryPool, 
                                  chaos: KigaiChaosEngine) -> np.ndarray:
        """
        Пакетная обработка 64-канального сигнала с учетом геометрического увода mu (coords) 
        и динамического Гауссовского сомнения. Вычисления выполняются за O(1).
        """
        # Извлекаем текущие гормональные маркеры с учетом чувствительности рецепторов
        cortisol = chemistry.cortisol
        serotonin = chemistry.serotonin
        dopamine = chemistry.dopamine
        
        # --- КОНТУР СОМНЕНИЯ И ГЕОМЕТРИЧЕСКОГО УВОДА (ФРИСТОН + ТЕЙЛОР) ---
        if cortisol > 0.1:
            # Генерируем вектор хаотического сомнения для каждого нейрона (64, 2)
            # Смещаем икс и игрек независимо через наш Тейлор-движок
            noise_matrix = np.array([[chaos.generate_quantum_noise(), chaos.generate_quantum_noise()] 
                                     for _ in range(self.size)])
            
            # Накладываем увод на координаты (наш бывший self.mu) пропорционально стрессу
            spatial_shift = noise_matrix * cortisol * 0.02
            self.coords = np.clip(self.coords + spatial_shift, 0.0, 1.0)
            
            # Выжигаем точность Гаусса (прецизионность) — растет сомнение сети
            # Используем среднее значение шума для скалярного подавления
            self.precision -= np.abs(np.mean(noise_matrix, axis=1)) * cortisol * 5.0
            self.precision = np.clip(self.precision, 0.1, 5.0)
            
        # Если стресса нет, серотонин медленно возвращает стабильность и затачивает точность
        else:
            self.precision = np.clip(self.precision + (serotonin * 0.05), 0.1, 5.0)
            
        # --- ВЫЧИСЛЕНИЕ АКТИВАЦИИ УНИВЕРСАЛЬНОЙ КОРЫ ---
        # Прямой поток: скалярное произведение входных весов на вектор данных
        excitation_in = np.dot(self.W_in, raw_vector)
        
        # Возвратный поток (Эхо): резонанс популяции. В будущем сюда зайдет матрица расстояний self.D
        excitation_rec = np.dot(self.W_rec, self.last_state)
        
        # Интегрируем потенциалы, масштабируя их на вектор индивидуальной точности нейронов
        total_input = (excitation_in + excitation_rec) * (self.precision / 2.0)
        
        # Выходной сигнал всей коры через векторизованную сигмоиду
        current_state = 1.0 / (1.0 + np.exp(-total_input))
        
                # --- ДИФФЕРЕНЦИАЛЬНАЯ ПЛАСТИЧНОСТЬ ХЕББА ---
        # 1. Находим производные (текущее состояние минус состояние на прошлом такте)
        # Для входа нам нужен сохраненный с прошлого шага raw_vector (добавь self.last_input в __init__)
        if not hasattr(self, 'last_input'):
            self.last_input = np.zeros(self.input_dim, dtype=np.float64)
            
        delta_input = raw_vector - self.last_input
        delta_output = current_state - self.last_state
        
        # 2. Модификатор шага пластичности с учетом выгорания дофаминовых рецепторов при стрессе
        plasticity_modifier = max(0.001, (self.learning_rate + dopamine * 0.02) * (1.0 - cortisol * 0.5))
        
        # 3. Дифференциальное обновление весов: веса растут только если знаки изменений совпали
        # Внешнее произведение производных выходов на производные входов
        self.W_in = np.clip(self.W_in + plasticity_modifier * np.outer(delta_output, delta_input), 0.0, 1.0)
        
        # Внутреннее возвратное обновление (закрепление семантических путей/эха)
        # Здесь мы берем изменение текущего выхода и изменение выхода на прошлом шаге
        if not hasattr(self, 'last_delta_output'):
            self.last_delta_output = np.zeros(self.size, dtype=np.float64)
            
        self.W_rec = np.clip(self.W_rec + plasticity_modifier * np.outer(delta_output, self.last_delta_output), 0.0, 0.5)
        np.fill_diagonal(self.W_rec, 0.0) # Жесткая защита от самовозбуждения
        
        # 4. Сохраняем исторические слои для дифференциала на следующем такте
        self.last_input = raw_vector.copy()
        self.last_delta_output = delta_output.copy()
        self.last_state = current_state.copy()
    
    
        return current_state
        