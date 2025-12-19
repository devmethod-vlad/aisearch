class DomainException(Exception):
    """Базовая ошибка предметной области"""


class TimeoutException(DomainException):
    """Ошибка частых запросов"""


class QueueMaxSizeException(DomainException):
    """Ошибка очереди"""


class TooManyRequestsException(DomainException):
    """Ошибка частых запросов"""


class FeedbackException(DomainException):
    """Ошибка обратной связи"""
