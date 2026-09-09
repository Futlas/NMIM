import json
import os
import random

import numpy as np


class KigaiSensorCortex:
    """
    Обновленная сенсорная кора ядра KIGAI.
    Нарезает текст на слоги с жестким порогом джиттера (>0.5),
    выделяет под каждый слог случайные 8 нейронов-адресатов
    и управляет их внутренними кубитными синапсами 8x8 через JSON.
    """

    def __init__(self, total_neurons: int = 100, Weights_path:str = "evo_genes.json"):
        
        self.total_neurons = total_neurons
        self.Weights_path = Weights_path
        self.sillywindow = 0

        self.alphabet = {}
        self.load_cluster_data()

    def load_cluster_data(self):
        if os.path.exists(self.Weights_path):
            with open(self.Weights_path, "r", encoding="utf-8") as f:
                try:
                    self.alphabet = json.load(f)
                except json.JSONDecodeError:
                    self.alphabet = {}
        else:
            self.alphabet = {}
    def save_cluster_data(self):

        with open(self.Weights_path, "w", encoding="utf-8") as f:
            json.dump(self.alphabet,f,ensure_ascii=False,indent=2)

    def get_or_create_cluster(self, syllable: str, neuron_coordinates: np.ndarray) -> dict:
        """
            Шаг 2: Произвольное рождение локального кластера. 
        Находит случайный свободный нейрон-центр и собирает 7 ближайших свободных соседей.
        """
        if syllable in self.alphabet:
            return self.alphabet[syllable]

              # 1. Сбор уже занятых нейронов (с защитой от старых данных в JSON)
        occupied_ids = set()
        for data in self.alphabet.values():
            # Проверяем, что внутри лежит словарь, а не число float
            if isinstance(data, dict) and "neuron_ids" in data:
                occupied_ids.update(data["neuron_ids"])


        all_ids = set(range(self.total_neurons))
        free_ids = list(all_ids - occupied_ids)

        # Если сеть забита, берем вообще любые свободные нейроны
        if len(free_ids) < 8:
            free_ids = list(all_ids)

        # 2. Выбираем случайную "точку сборки" (первый свободный нейрон)
        center_id = random.choice(free_ids)
        
        # 3. Ищем ближайших соседей по геометрии Кулона
        center_coords = neuron_coordinates[center_id]
        
        # Считаем расстояния от центра до всех остальных свободных нейронов
        distances = []
        for f_id in free_ids:
            if f_id != center_id:
                # Обычное евклидово расстояние между точками (x1-x2)^2 + (y1-y2)^2
                dist = np.sum((neuron_coordinates[f_id] - center_coords) ** 2)
                distances.append((dist, f_id))
        
        # Сортируем по близости и забираем 7 самых близких соседей
        distances.sort(key=lambda x: x[0])
        closest_neighbors = [item[1] for item in distances[:7]]

        # Финальный ансамбль: центр + 7 ближайших точек
        chosen_ids = [center_id] + closest_neighbors

        # 4. Строим матрицу кубитов 8x8 (суперпозиция 0.5)
        synapse_matrix = np.full((8, 8), 0.5).tolist()

        # Записываем в память и сохраняем в JSON
        self.alphabet[syllable] = {
            "neuron_ids": chosen_ids,
            "synapse_matrix": synapse_matrix
        }
        self.save_cluster_data()
        
        return self.alphabet[syllable]
    
    def step_tokenize(self, text: str, cortisol: float, neuron_coordinates: np.ndarray) -> tuple: 
        """
        Шаг 3: Тактовая нарезка.
        Вырезает слог с учетом жесткого порога стресса, находит его компактный 
        геометрический кластер и выдает адреса нейронов и интенсивность (0-7).
        """
        if not text:
            # Если текста нет, возвращаем пустые массивы
            return np.zeros(8, dtype=np.int32), np.zeros(8, dtype=np.int32)
        
        # Если дошли до конца текста, сбрасываем окно в начало
        if self.sillywindow >= len(text):
            self.sillywindow = 0

        # Твой жесткий пороговый триггер джиттера (Шаг 1)
        base_window = 2
        if cortisol > 0.5:
            # Стресс пробил порог — врубаем хаос-окно со сдвигом +1
            window_size = base_window + 1
        else:
            window_size = base_window

        # Вырезаем слог из текста
        chunk = text[self.sillywindow : self.sillywindow + window_size]
        self.sillywindow += len(chunk)

        # Передаем координаты нейронов для поиска ближайших соседей (Шаг 2)
        cluster = self.get_or_create_cluster(chunk, neuron_coordinates)
        neuron_ids = np.array(cluster["neuron_ids"], dtype=np.int32)

        # Базовая максимальная интенсивность сигнала для 8 каналов
        intensities = np.full(8, 7, dtype=np.int32)

        # Если сеть в стрессе, хаос-движок слегка тушит сигнал на случайных каналах
        if cortisol > 0.3:
            quantum_sample = self.chaos.generate_quantum_noise()
            intensities = np.clip(intensities - int(quantum_sample * 3), 0, 7)

        # Возвращаем кортеж: (ID 8 нейронов, Сила импульса от 0 до 7)
        return neuron_ids, intensities

if __name__ == "__main__":
    print("=== ЗАПУСК ТОПОЛОГИЧЕСКОГО ТЕСТА КОРЫ ===")
    
    # Генерируем тестовые координаты для 100 нейронов
    np.random.seed(42)
    fake_coordinates = np.random.uniform(-10.0, 10.0, (100, 2))
    
    # Создаем кору с привязкой к evo_genes.json
    cortex = KigaiSensorCortex(total_neurons=100, Weights_path="evo_genes.json")
    
    test_text = "ПАША ПАША"
    print(f"Входной текст: '{test_text}'")
    
    # Имитируем такты чтения текста
    ids_1, int_1 = cortex.step_tokenize(test_text, cortisol=0.0, neuron_coordinates=fake_coordinates)
    print(f"\nТакт 1 | Нейроны: {ids_1} | Интенсивность: {int_1}")
    
    ids_2, int_2 = cortex.step_tokenize(test_text, cortisol=0.0, neuron_coordinates=fake_coordinates)
    print(f"Такт 2 | Нейроны: {ids_2} | Интенсивность: {int_2}")
