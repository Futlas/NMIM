import asyncio
import json
import os
import random

import numpy as np
import uvloop

from chaos_engine import KigaiChaosEngine

# Импортируем модули нашего изолированного ядра NMIM
from chemistry_pool import KigaiChemistryPool
from sensor_cortex import KigaiSensorCortex
from virtual_neuron import KigaiPopulationLayer

# Принудительно заводим uvloop под CachyOS
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Статические пути к памяти гомеостаза на SSD
WEIGHTS_PATH = "kigai_weights.json"
EVO_PATH = "evo_genes.json"

def load_genes() -> dict:
    """Загружает эволюционные гены гомеостаза или возвращает дефолтный генотип MVP."""
    if os.path.exists(EVO_PATH):
        try:
            with open(EVO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:  # noqa: BLE001, S110
            pass
    # Дефолтная ДНК для первого старта (базовые коэффициенты)
    return {
        "cortisol_shock": 0.10,
        "dopamine_burn": -0.03,
        "adrenaline_serotonin": 0.85,
        "best_fitness": -999.0,
        "generation": 0,
        "repulsion_base": 0.02
    }

def save_genes(genes: dict):
    """Безопасно сохраняет выигрышные гены на SSD."""
    try:
        with open(EVO_PATH, "w", encoding="utf-8") as f:
            json.dump(genes, f, indent=4, ensure_ascii=False)
    except Exception:  # noqa: BLE001, S110
        pass

def mutate_genes(base_genes: dict) -> dict:
    """Применяет случайную мутацию к генам (Эволюционное блуждание Тейлора)."""
    mutated = base_genes.copy()
    scale = 0.15
    
    # Мутируем гены в жестких биологических границах
    current_repulsion = base_genes.get("repulsion_base", 0.02)
    mutated["repulsion_base"] = np.clip(current_repulsion + random.uniform(-0.01, 0.01), 0.005, 0.05)
    mutated["cortisol_shock"] = np.clip(base_genes["cortisol_shock"] + random.uniform(-scale, scale), 0.01, 0.4)
    mutated["dopamine_burn"] = np.clip(base_genes["dopamine_burn"] + random.uniform(-scale, scale), -0.15, -0.005)
    mutated["adrenaline_serotonin"] = np.clip(base_genes["adrenaline_serotonin"] + random.uniform(-scale, scale), 0.3, 1.0)
    mutated["generation"] += 1
    return mutated

async def run_testing_epoch(genes: dict, cortex: KigaiSensorCortex):
    """Прогоняет жесткий тест: 10 изолированных сессий по 100 тактов."""
    total_actions = 0
    total_cortisol_accumulation = 0.0
    total_ticks = 0
    
    # Наш тренировочный пул предложений для проверки инвариантности смысла
    env_sentences = [
        "Паша ищет дзен",
        "Ищет Паша дзен",
        "Дзен ищет Паша"
    ]
    
    # ВНЕШНИЙ ЦИКЛ: 10 сессий для чистой накопленной статистики выживания
    for session in range(1, 11):
        chemistry = KigaiChemistryPool()
        chaos = KigaiChaosEngine()
        neuron = KigaiPopulationLayer()
        
        # Подгружаем сохраненную синаптическую память Хебба прошлых поколений
        if os.path.exists(WEIGHTS_PATH):
            try:
                with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:  # noqa: ASYNC230
                    neuron.import_weights(json.load(f).get("neuron", {}))
            except Exception:  # noqa: BLE001, S110
                pass

        # Инициализируем статическую матрицу памяти предложений ОДИН раз за сессию.
        # Чтобы не сбивать sillywindow живого кортекса, генерируем её через независимый чистый запуск
        temp_cortex = KigaiSensorCortex(channels_count=cortex.channels_count)
        memory_matrices = np.array([
            temp_cortex.step_tokenize(s, 0.0, chaos)
            for s in env_sentences
        ])

        prediction_error = 0.0
        current_sentence = random.choice(env_sentences)
        
        # ВНУТРЕННИЙ ЦИКЛ: Ровно 100 тактов Closed-Loop гомеостаза
        for tick_count in range(1, 101):
            total_ticks += 1
            
            # Если токенизатор дочитал предложение до конца (sillywindow сбросился в 0),
            # или это самый первый такт — выбираем новое случайное предложение-инверсию
            if cortex.sillywindow == 0 or tick_count == 1:
                current_sentence = random.choice(env_sentences)
                current_prediction_error = 0.5
            else:
                current_prediction_error = prediction_error

            # Нарезаем текущее предложение на адаптивные слоги
            raw_sensory_input = cortex.step_tokenize(current_sentence, chemistry.cortisol, chaos)
            raw_sensory_input = np.clip(raw_sensory_input * 1.6, 0.0, 1.0)

            chemistry.update_metabolism(current_prediction_error)
            neuron.update_spatial_topology(repulsion_base=genes.get("repulsion_base", 0.02))

            # ГЕН 1: Аварийный адреналиновый форсаж серотонина
            if chemistry.cortisol >= 0.85:
                chemistry.modify_hormone(chemistry.IDX_SEROTONIN, genes["adrenaline_serotonin"])
                
            # Твой контур слива излишков серотонина
            if chemistry.serotonin >= 0.90 and chemistry.cortisol <= (chemistry.serotonin / 2):
                chemistry.modify_hormone(chemistry.IDX_SEROTONIN, -0.10)
                
            # --- ОБРАБОТКА ПОПУЛЯЦИЕЙ ---
            population_state = neuron.process_population_signal(raw_sensory_input, chemistry, chaos)
            
            # Физическое движение графа коры на каждом такте
            neuron.update_spatial_topology(repulsion_base=genes.get("repulsion_base", 0.02))
            
            # Динамический порог принятия решений (сужается серотонином)
            dynamic_threshold = 0.6 - (chemistry.serotonin * 0.15)
            mean_activation = float(np.mean(population_state))
            
            if mean_activation > dynamic_threshold:
                # МАТРИЧНЫЙ МОТОРНЫЙ АКТ (Active Inference)
                total_actions += 1

                # Евклидово расстояние от текущего слогового спайка до паттернов предложений
                distances = np.linalg.norm(memory_matrices - raw_sensory_input, axis=1)
                best_sentence_idx = np.argmin(distances)
                predicted_sentence = env_sentences[best_sentence_idx]
                
                # Если модель по текущему слогу смогла правильно распознать, 
                # какое именно предложение сейчас читается — это УСПЕХ
                if predicted_sentence == current_sentence:
                    prediction_error = max(0.0, current_prediction_error - 0.5)
                    chemistry.modify_hormone(chemistry.IDX_DOPAMINE, 0.30)
                    chemistry.modify_hormone(chemistry.IDX_SEROTONIN, 0.20)
                else:
                    # ОШИБКА -> Кортизоловый шок
                    prediction_error = min(1.0, current_prediction_error + 0.3)
                    chemistry.modify_hormone(chemistry.IDX_CORTISOL, 0.25)
                    chemistry.modify_hormone(chemistry.IDX_DOPAMINE, -0.10)
            else:
                # ПАССИВНОЕ НАБЛЮДЕНИЕ (Обдумывание)
                prediction_error = current_prediction_error
                chemistry.modify_hormone(chemistry.IDX_CORTISOL, genes["cortisol_shock"])
                chemistry.modify_hormone(chemistry.IDX_DOPAMINE, genes["dopamine_burn"])

            total_cortisol_accumulation += chemistry.cortisol

        # На 100-м такте сессии жестко пишем веса Хебба на диск перед дропом гомеостаза
        try:
            with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:  # noqa: ASYNC230
                json.dump({"neuron": neuron.export_weights()}, f, indent=4)
        except Exception:  # noqa: BLE001, S110
            pass

    # --- ТВОЙ ОРИГИНАЛЬНЫЙ ФИНАЛ И КРИТЕРИЙ ВЫЖИВАНИЯ ---
    # Вычисляем коэффициент выживаемости (Fitness Score) за всю эпоху
    mean_cortisol = total_cortisol_accumulation / total_ticks
    fitness = float(total_actions) - (mean_cortisol * 10.0)
    
    # Считаем среднее пространственное расстояние между нейронами последнего поколения
    mean_spatial_dist = np.mean(neuron.D)
    
    # Если нейроны схлопнулись в точку (<0.15) или разлетелись врассыпную (>0.8) — жестко штрафуем фитнес
    if mean_spatial_dist < 0.15 or mean_spatial_dist > 0.8: 
        fitness -= 300.0
        
    return fitness, total_actions, mean_cortisol


async def main_evolution_loop():
    print("="*60)
    print(" ЗАПУСК ЭВОЛЮЦИОННОГО ПОЛИГОНА НМИМ (v0.2-EVO)")
    print("="*60)
    
    cortex = KigaiSensorCortex(channels_count=64)
    current_genes = load_genes()
    
    print(f"[ЭВОЛЮЦИЯ] Старт с поколения: ГЕН-{current_genes['generation']}")
    print(f"[ЭВОЛЮЦИЯ] Текущий рекорд Fitness: {current_genes['best_fitness']:.2f}\n")
    
    while True:
        # Оценка базового генотипа при самом первом прогоне
        if current_genes["best_fitness"] == -999.0:
            print("[ОТБОР] Сбор статистики базового генотипа...")
            fit, acts, cort = await run_testing_epoch(current_genes, cortex)
            current_genes["best_fitness"] = fit
            save_genes(current_genes)
            print(f"-> Результат базы: Fitness={fit:.2f} | Ходов={acts} | Ср.Кортизол={cort:.3f}\n")
            
        # Рождаем мутанта через случайное Тейлор-блуждание
        mutant_genes = mutate_genes(current_genes)
        print(f"[ПОКОЛЕНИЕ {mutant_genes['generation']}] Мутация параметров метаболизма:")
        print(f"   Шок Кортизола (else)       : {mutant_genes['cortisol_shock']:.4f}")
        print(f"   Слив Дофамина (else)       : {mutant_genes['dopamine_burn']:.4f}")
        print(f"   Вброс Серотонина (кризис)   : {mutant_genes['adrenaline_serotonin']:.4f}")
        
        # Запускаем отбор мутанта на дистанции 1000 тактов (10 по 100)
        print("   [ИНКУБАТОР] Тестирование 10 сессий по 100 тактов...")
        mutant_fit, mutant_acts, mutant_cort = await run_testing_epoch(mutant_genes, cortex)
        print(f"   [ИТОГ] Результат мутанта: Fitness={mutant_fit:.2f} | Ходов={mutant_acts} | Ср.Кортизол={mutant_cort:.3f}")
        
        # ЕСТЕСТВЕННЫЙ ОТБОР НМИМ
        if mutant_fit > current_genes["best_fitness"]:
            print(f" ЭВОЛЮЦИОННЫЙ ПРОРЫВ! Продуктивность выросла на +{mutant_fit - current_genes['best_fitness']:.2f}!")
            print("   Старая ДНК вытеснена. Новый успешный генотип записан на SSD.\n")
            current_genes = mutant_genes
            current_genes["best_fitness"] = mutant_fit
            save_genes(current_genes)
        else:
            print(" Мутация признана нежизнеспособной (сеть ушла в кому). Откат к стабильному предку.\n")
            
        await asyncio.sleep(0.5)  # Тактовая пауза между поколениями для наглядности логов

if __name__ == "__main__":
    asyncio.run(main_evolution_loop())
