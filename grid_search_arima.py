"""
Класс, реализующий поиск параметров по сетке для модели ARMA/ARIMA. Нужен для
задания юнита 4 ("4. Статистические модели прогнозирования").
"""

from itertools import product
from pprint import pformat
from typing import Dict, Iterable, Optional, Type
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import NotFittedError
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base.prediction import PredictionResults

PARAMS = ["p", "q"]

# предупреждения, указывающие на то, что AIC модели должен быть inf
WARNINGS = [
    "Non-stationary starting autoregressive parameters found.",
    "Non-invertible starting MA parameters found."
]


class GridSearch:
    """
    Перебор параметров p, d и q по сетке для модели ARMA/ARIMA.

    Attrs:
        __estimator: тип модели (ARMA или ARIMA)
        __param_grid: сетка параметров
        results_: словарь с результатами перебора параметров
        best_estimator_: наилучшая модель
        best_score_: значение AIC для наилучшей модели
        best_params_: наилучшие параметры модели
        best_index_: индекс наилучших параметров
    """

    def __init__(
        self,
        estimator: Type[ARMA | ARIMA],
        param_grid: Dict[str, Iterable[int]],
    ):
        """
        Инициализирует объект класса.

        Args:
            estimator: модель ARMA/ARIMA
            param_grid: сетка параметров (p и q для ARMA, p, d и q для ARIMA)

        Raises:
            ValueError, если estimator не является моделью ARMA/ARIMA
            ValueError, если сетка параметров не соответствует модели
        """
        if estimator not in [ARMA, ARIMA]:
            raise ValueError("Model must be either ARMA or ARIMA")

        self.__estimator = estimator

        if set(param_grid.keys()) != set(PARAMS):
            raise ValueError("Parameters must be p and q")

        self.__param_grid = param_grid
        self.results_ = {
            "param_p": [],
            "param_q": [],
            "score": [],
            "params": [],
        }
        self.best_estimator_ = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None

    def __repr__(self):
        return (
            f"GridSearch(estimator={self.__estimator}, "
            f"param_grid={pformat(self.__param_grid)})"
        )

    def __str__(self):
        return (
            f"GridSearch(estimator={self.__estimator}, "
            f"param_grid={pformat(self.__param_grid)})"
        )

    def get_params(self) -> Dict[str, Iterable[int]]:
        """
        Возвращает сетку параметров.

        Returns:
            сетка параметров
        """
        return self.__param_grid

    def set_params(self, param_grid: Dict[str, Iterable[int]]):
        """
        Устанавливает новую сетку параметров.

        Returns:
            сетка параметров
        """
        self.__param_grid = param_grid

    def check_is_fitted(self):
        """
        Проверяет, найдена ли наилучшая модель.

        Raises:
            NotFittedError, если наилучшая модель не найдена
        """
        if not self.best_estimator_:
            raise NotFittedError(msg % {"name": self.__estimator.__name__})

    def fit(self, X: pd.Series, d: Optional[int]):
        """
        Для заданного временного ряда овершает перебор параметров по сетке и
        находит наилучшую модель.

        Args:
            X: временной ряд
            d: значение параметра d (только для модели ARIMA)
        """
        orders = product(*[self.__param_grid[param] for param in PARAMS])

        if self.__estimator == ARIMA:
            orders = [(p, d, q) for p, q in orders]

        for i, order in tqdm(enumerate(orders)):
            with catch_warnings(record=True) as record:
                simplefilter("always")
                estimator = self.__estimator(X, order=order).fit()
                score = np.inf if any([any([
                    message in str(warning.message) for message in WARNINGS
                ]) for warning in record]) else estimator.aic
            params = {
                "p": order[0],
                "q": order[-1],
            }
            if self.__estimator == ARIMA:
                params["d"] = d
            if score != np.inf and (
                self.best_estimator_ is None or score < self.best_score_
            ):
                self.best_estimator_ = estimator
                self.best_score_ = score
                self.best_params_ = order
                self.best_index_ = i
            self.results_["param_p"].append(params["p"])
            self.results_["param_q"].append(params["q"])
            self.results_["score"].append(score)
            self.results_["params"].append(params)

    def summary(self):
        """
        Выводит на экран саммари наилучшей модели.

        Raises:
            NotFittedError, если наилучшая модель не найдена
        """
        self.check_is_fitted()
        self.best_estimator_.summary()

    def predict(self, X: pd.Series) -> pd.Series:
        """
        Делает предсказания для временного ряда с помощью наилучшей модели.

        Args:
            X: временной ряд

        Returns:
            предсказания для временного ряда

        Raises:
            NotFittedError, если наилучшая модель не найдена
        """
        self.check_is_fitted()
        return self.best_estimator_.predict(start=X.index[0], end=X.index[-1])

    def get_forecast(self, X: pd.Series, **kwargs) -> PredictionResults:
        """
        Делает прогноз для временного ряда с помощью наилучшей модели.

        Args:
            X: временной ряд
            kwargs: ключевые аргументы для метода get_forecast модели

        Returns:
            прогноз для временного ряда

        Raises:
            NotFittedError, если наилучшая модель не найдена
        """
        self.check_is_fitted()
        return self.best_estimator_.get_forecast(len(X.index), **kwargs)
