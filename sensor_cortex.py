import numpy as np


class KigaiSensorCortex:

    """
    Сенсорная кора ядра KIGAI.
    Парсит текстовые сигналы среды и транслирует их в гладкие 
    пространственные векторы волновых амплитуд [0.0, 1.0] для cl-sdk.
    """

    def __init__(self,channels_count: int = 64):

        self.channels_count = channels_count
    def text_to_wave_vector(self,text: str) -> np.ndarray:
        """
        Преобразует строку текста в монолитный вектор амплитуд вероятностей.
        Чистая интерполяция пространственного буфера.
        """

        if not text:
            return np.zeros(self.channels_count,dtype=np.float64)

        raw_ascii = np.array([ord(char) for char in text],dtype=np.float64)
        normalized_wave = (raw_ascii % 32) / 32.0

        x_old = np.linspace(0,1,len(normalized_wave))
        x_new = np.linspace(0,1,self.channels_count)

        spartial_pattern = np.interp(x_new,x_old,normalized_wave)

        return np.clip(spartial_pattern,0.0,1.0)

        
