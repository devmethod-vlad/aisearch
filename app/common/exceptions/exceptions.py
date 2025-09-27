class LoginError(Exception):
    """Ошибка авторизации"""

    pass


class NotFoundError(Exception):
    """Объект не найден"""

    pass


class MilvusCollectionLoadTimeoutError(Exception):
    """Ошибка времени ожидания загрузки коллекции"""

    pass


class MilvusDeletionTimeoutError(Exception):
    """Ошибка ожидания удаления коллекци"""


class RequestError(Exception):
    """Ошибка запроса"""

    pass
