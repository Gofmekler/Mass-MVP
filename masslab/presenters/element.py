from masslab.model.elements import ELEMENTS
from masslab.model.tasks import MASS_TOLERANCE, ElementTask
from masslab.presenters.formatting import num, parse_number, reference_rows
from masslab.presenters.workspace import WorkspacePresenter


class ElementPresenter:
    """Задание 2: определить массу и элемент неизвестного иона."""

    def __init__(self, view, rng, np_rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.02)
        self._task = None
        self._attempts = 0
        view.check_requested.connect(self._on_check)

    def start(self):
        self._task = ElementTask(self._rng)
        self._attempts = 0
        self._ws.reset()
        self._view.show_reference(reference_rows())
        self._show_sample()
        self._view.show_feedback("", None)

    def _show_sample(self):
        self._view.clear_inputs()
        self._view.set_options([f"{s} — {ELEMENTS[s].name}" for s in self._task.options])
        self._ws.set_peaks(self._task.peaks())

    def _on_check(self):
        mass = parse_number(self._view.mass_text())
        if mass is None:
            self._view.show_feedback("Введите массу неизвестного иона числом, например 22,9.",
                                     False)
            return
        option = self._view.selected_option()
        if option is None:
            self._view.show_feedback("Выберите элемент.", False)
            return
        self._attempts += 1
        result = self._task.check(mass, option)
        if result.passed:
            self._view.show_feedback("Верно! Задание 2 выполнено.", True)
            self._on_completed(self._attempts)
            return
        if not result.mass_ok and not result.element_ok:
            reason = "Неверно определены и масса, и элемент."
        elif not result.mass_ok:
            reason = f"Масса определена неточно (допуск ±{MASS_TOLERANCE * 100:.0f} %)."
        else:
            reason = "Элемент выбран неверно."
        e = self._task.unknown
        self._task.new_sample()
        self._show_sample()
        self._view.show_feedback(
            f"{reason} Правильный ответ: {e.symbol} — {e.name}, {num(e.mass, 2)} а.е.м. "
            "Выдан новый образец.", False)
