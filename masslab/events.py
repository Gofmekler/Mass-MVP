"""Простое событие без зависимости от Qt — для связи пассивных видов с презентерами."""


class Event:
    def __init__(self):
        self._handlers = []

    def connect(self, handler):
        self._handlers.append(handler)

    def emit(self, *args):
        for handler in list(self._handlers):
            handler(*args)
