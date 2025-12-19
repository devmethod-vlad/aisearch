import typing as tp
from abc import ABC, abstractmethod


class IHealthCheck(ABC):
    @abstractmethod
    async def check(self) -> dict[str, tp.Any]:
        """Выполнить проверку здоровья"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя проверки"""
        pass
