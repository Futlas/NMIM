import numpy as np


class KigaiChemistryPool:
    """
    Модуль низкоуровневого метаболизма ядра KIGAI (v0.1-silico).
    Выделяет монолитный плоский массив на 128 ячеек float64 для L1-Cache локализации.
    Реализует функциональные классы гормонов и детерминированный half-life распад.
    """
    def __init__(self):
        # Выделяем ровно 1 Кб непрерывной памяти (128 элементов * 8 байт)
        self._pool = np.zeros(128, dtype=np.float64)
        
        # Статические индексы объединенных функциональных классов гормонов
        self.IDX_DOPAMINE = 0   # Класс: Закрепление успешного паттерна / Фокус
        self.IDX_CORTISOL = 1   # Класс: Стресс / Высокочастотный хаос
        self.IDX_SEROTONIN = 2  # Класс: Торможение хаоса / Активное сопротивление
        self.IDX_GLUCOSE = 3    # Класс: Энергетический субстрат (АТФ)
        
        # Базовый стартовый гомеостаз
        self._pool[self.IDX_GLUCOSE] = 1.0  # Максимум энергии на старте
        self._pool[self.IDX_DOPAMINE] = 0.1  # Небольшой фоновый дофамин
        
        # Ежетактовые коэффициенты half-life деградации (KISS)
        self.DECAY_CORTISOL = 0.05   # Базовый распад -5% за шаг
        self.DECAY_DOPAMINE = 0.10   # -10% за шаг
        self.DECAY_SEROTONIN = 0.08  # -8% за шаг

    def update_metabolism(self, prediction_error: float):
        """
        Ежетактовое обновление гормонов.
        МЕГА-АПГРЕЙД: Серотонин динамически увеличивает скорость распада кортизола!
        """
        # Динамический half-life: если серотонин на пике, кортизол испаряется в 5 раз быстрее!
        dynamic_cortisol_decay = self.DECAY_CORTISOL + (self._pool[self.IDX_SEROTONIN] * 0.25)
        
        # Применяем распад
        self._pool[self.IDX_CORTISOL] *= (1.0 - dynamic_cortisol_decay)
        self._pool[self.IDX_DOPAMINE] *= (1.0 - self.DECAY_DOPAMINE)
        self._pool[self.IDX_SEROTONIN] *= (1.0 - self.DECAY_SEROTONIN)
        self._pool[self.IDX_GLUCOSE] -= 0.01  # Линейный расход АТФ
        
        # Контур перекрестного влияния ошибки предсказания
        if prediction_error > 0.1:
            self._pool[self.IDX_CORTISOL] += prediction_error * 0.2
            if self._pool[self.IDX_SEROTONIN] > 0.5:
                self._pool[self.IDX_CORTISOL] -= self._pool[self.IDX_SEROTONIN] * 0.1
        else:
            self._pool[self.IDX_DOPAMINE] += 0.05
            self._pool[self.IDX_SEROTONIN] += 0.02

        self._pool = np.clip(self._pool, 0.0, 1.0)

    def modify_hormone(self, index: int, amount: float):
        """
        Универсальный контур изменения концентрации гормонов.
        Принимает как положительные (впрыск), так и отрицательные (откачка) значения.
        Безопасно удерживает баланс внутри Марковского одеяла [0.0, 1.0].
        """
        if 0 <= index < 128:
            new_value = self._pool[index] + amount
            self._pool[index] = np.clip(new_value, 0.0, 1.0)

    # Удобные свойства (property) для быстрого чтения параметров другими модулями
    @property
    def cortisol(self) -> float:
        return self._pool[self.IDX_CORTISOL]

    @property
    def dopamine(self) -> float:
        return self._pool[self.IDX_DOPAMINE]

    @property
    def serotonin(self) -> float:
        return self._pool[self.IDX_SEROTONIN]

    @property
    def glucose(self) -> float:
        return self._pool[self.IDX_GLUCOSE]
