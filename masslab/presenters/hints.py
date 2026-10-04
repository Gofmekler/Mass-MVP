FAILURES_BEFORE_HINT = 2


class HintTracker:
    """Показывает подсказки после нескольких неудачных попыток подряд."""

    def __init__(self, show_hint, hints):
        self._show = show_hint
        self._hints = hints
        self._streak = 0

    def reset(self):
        self._streak = 0
        self._show(None)

    def failed(self):
        self._streak += 1
        if self._streak >= FAILURES_BEFORE_HINT:
            index = min(self._streak - FAILURES_BEFORE_HINT, len(self._hints) - 1)
            self._show(f"Подсказка {index + 1} из {len(self._hints)}. {self._hints[index]}")
