class LoginError(Exception):
    """Ошибка авторизации"""

    pass


class NotFoundError(Exception):
    """Объект не найден"""

    pass


class MilvusCollectionLoadTimeoutError(Exception):
    """Ошибка времени ожидания загрузки коллекции"""

    pass


class RequestError(Exception):
    """Ошибка запроса"""

    pass
