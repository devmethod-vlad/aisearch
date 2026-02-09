class LoginError(Exception):
    """Ошибка авторизации"""

    pass


class NotFoundError(Exception):
    """Объект не найден"""

    pass


class AlreadyExistsError(Exception):
    """Ошибка наличествования"""

    pass


class ConflictError(Exception):
    """Конфликт запроса"""

    pass


class MilvusCollectionLoadTimeoutError(Exception):
    """Ошибка времени ожидания загрузки коллекции"""

    pass


class MilvusDeletionTimeoutError(Exception):
    """Ошибка ожидания удаления коллекци"""


class RequestException(Exception):
    """Ошибка запроса"""

    pass


class NoCudaException(Exception):
    """Ошибка отсутствия CUDA"""

    pass
