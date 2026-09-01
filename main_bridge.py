import asyncio
import random

import numpy as np
import uvloop

from chaos_engine import KigaiChaosEngine

# Наше ядро
from chemistry_pool import KigaiChemistryPool
from sensor_cortex import KigaiSensorCortex
from virtual_neuron import VirtualBioNeuron

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def running_incubator_loop():
    print("[KIGAI] Инициализация лингвистических интерфейсов...")
    
    chemistry = KigaiChemistryPool()
    chaos = KigaiChaosEngine()
    neuron = VirtualBioNeuron()
    cortex = KigaiSensorCortex(channels_count=64)
    
    # Словарик среды (из него алгоритм кидает случайные слова для теста)
    env_dictionary = ["Москва", "Архангельск", "Астрахань", "Новгород", "Донецк", "Казань"]
    
    # Собственный словарик памяти нейрона (слова, которые он умеет говорить)
    neuron_memory = ["Астрахань", "Новгород", "Донецк", "Казань", "Моршанск", "Владивосток"]
    
    prediction_error = 0.0
    tick_count = 0
    last_env_word = ""
    
    # Статистика бенчмарка
    max_dopamine = max_cortisol = max_serotonin = max_output = 0.0
    active_actions_count = 0
    
    print("[KIGAI] Лингвистическая песочница 'Города без правил' запущена.\n")
    
    try:
        while True:
            tick_count += 1
            
            # --- ПРЯМОЙ ПОТОК (Среда делает ход) ---
            # Каждые 5 тактов среда вбрасывает новое случайное слово из словарика
            if tick_count % 5 == 0 or last_env_word == "":
                text_signal = random.choice(env_dictionary)
                print(f"\n[Такт {tick_count}] Среда сделала ход: '{text_signal}'")
                
                # Если это не первый ход, проверяем, совпала ли прошлая реакция сети с буквой среды
                if last_env_word:
                    # Упрощенная проверка стыковки букв (убираем регистр)
                    expected_letter = last_env_word[-1].lower()
                    if expected_letter in ['ь', 'ъ', 'ы']:  # Правило городов для граничных букв
                        expected_letter = last_env_word[-2].lower()
                last_env_word = text_signal
                current_prediction_error = 0.5  # Базовый шок от нового слова среды
            else:
                text_signal = "HEARTBEAT_STABLE"
                current_prediction_error = prediction_error

            # Сенсорная кора превращает слово в ASCII волновой вектор
            wave_vector = cortex.text_to_wave_vector(text_signal)
            raw_sensory_input = float(np.mean(wave_vector)) * 1.6
            raw_sensory_input = np.clip(raw_sensory_input, 0.0, 1.0)
            
            # Считаем гормоны
            chemistry.update_metabolism(current_prediction_error)
            
            # Аварийный серотонин при глубоком стрессе
            if chemistry.cortisol >= 0.85:
                chemistry.modify_hormone(chemistry.IDX_SEROTONIN, 0.85)
                
            # Негативная обратная связь (Твой контур слива серотонина)
            if chemistry.serotonin >= 0.90 and chemistry.cortisol <= (chemistry.serotonin / 2):
                chemistry.modify_hormone(chemistry.IDX_SEROTONIN, -0.10)
                
            # Нейрон считает выходную волну
            output_action_probability = neuron.process_signal(raw_sensory_input, chemistry, chaos)
            
            # --- ОБРАТНЫЙ ПОТОК (Сеть делает свой ход через Active Inference) ---
            dynamic_threshold = 0.6 - (chemistry.serotonin * 0.15)
            
            if output_action_probability > dynamic_threshold:
                # Сеть пробивает ступор и извлекает слово из памяти!
                # Переводим слова памяти через кору и ищем то, чья плотность ближе всего к матожиданию синапса mu
                best_word = neuron_memory[0]
                min_dist = 999.0
                
                for word in neuron_memory:
                    word_density = float(np.mean(cortex.text_to_wave_vector(word)))
                    dist = abs(word_density - neuron.mu)
                    if dist < min_dist:
                        min_dist = dist
                        best_word = word
                
                action_status = f"ОТВЕТ СЕТИ: '{best_word}'"
                active_actions_count += 1
                
                # СЕМАНТИЧЕСКИЙ СУДЬЯ (Проверяем стыковку без учителей)
                # Последняя рабочая буква слова среды должна совпадать с первой буквой ответа сети
                target_letter = last_env_word[-1].lower()
                if target_letter in ['ь', 'ъ', 'ы']:
                    target_letter = last_env_word[-2].lower()
                    
                            # Вычисляем ASCII-плотность слова, которое выбрала сеть
            chosen_word_density = float(np.mean(cortex.text_to_wave_vector(best_word)))

            if best_word.lower().startswith(target_letter):
                # ПРАВИЛЬНЫЙ ОТВЕТ -> Резонанс. Притягиваем mu к плотности этого слова (закрепляем успех)
                action_status += " -> [РЕЗОНАНС БУКВ (+Дофамин)]"
                prediction_error = max(0.0, current_prediction_error - 0.5)
                chemistry.modify_hormone(chemistry.IDX_DOPAMINE, 0.30)
                chemistry.modify_hormone(chemistry.IDX_SEROTONIN, 0.20)
                
                # Пластичность Хебба: mu плавно центрируется на успешном слове
                neuron.mu = 0.8 * neuron.mu + 0.2 * chosen_word_density
            else:
                # НЕПРАВИЛЬНЫЙ ОТВЕТ -> Кортизоловая боль. 
                action_status += " -> [НЕЛИСТОВЫЙ СДВИГ (+Кортизол)]"
                prediction_error = min(1.0, current_prediction_error + 0.3)
                chemistry.modify_hormone(chemistry.IDX_CORTISOL, 0.25)
                chemistry.modify_hormone(chemistry.IDX_DOPAMINE, -0.10)
                
                # ПЛАСТИЧНОСТЬ ОТЧАЯНИЯ: Кортизол пинком отшвыривает mu от этого слова!
                # Сеть физически больше не сможет выбрать "Донецк" на следующем шаге
                if neuron.mu > chosen_word_density:
                    neuron.mu = min(1.0, neuron.mu + 0.25 * chemistry.cortisol)
                else:
                    neuron.mu = max(0.0, neuron.mu - 0.25 * chemistry.cortisol)

            # Собираем рекорды
            max_dopamine = max(max_dopamine, chemistry.dopamine)
            max_cortisol = max(max_cortisol, chemistry.cortisol)
            max_serotonin = max(max_serotonin, chemistry.serotonin)
            max_output = max(max_output, output_action_probability)

            print(
                f"Лог -> Дофамин: {chemistry.dopamine:.3f} | "
                f"Кортизол: {chemistry.cortisol:.3f} | "
                f"Серотонин: {chemistry.serotonin:.3f} | "
                f"{action_status}"
            )
            
            await asyncio.sleep(0.05)
            
            # Сброс 100 тактов
            if tick_count >= 100:
                print("\n" + "="*60)
                print(" ЛИНГВИСТИЧЕСКИЕ ИТОГИ СЕССИИ (100 ТАКТОВ):")
                print("="*60)
                print(f" Пиковый Дофамин  : {max_dopamine:.3f}")
                print(f" Пиковый Кортизол : {max_cortisol:.3f}")
                print(f" Пиковый Серотонин: {max_serotonin:.3f}")
                print(f" Успешных ходов сети: {active_actions_count}")
                print("="*60)
                print("[KIGAI] Автоматический дроп и перезапуск лингвистического гомеостаза...\n")
                
                chemistry = KigaiChemistryPool()
                chaos = KigaiChaosEngine()
                neuron = VirtualBioNeuron()
                prediction_error = 0.0
                tick_count = 0
                max_dopamine = max_cortisol = max_serotonin = max_output = 0.0
                active_actions_count = 0
                last_env_word = ""
                await asyncio.sleep(2.0)
                
    except asyncio.CancelledError:
        print("[KIGAI] Асинхронный цикл остановлен.")

if __name__ == "__main__":
    asyncio.run(running_incubator_loop())
