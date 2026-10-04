from masslab.model.tasks import AlloyTask
from masslab.presenters.formatting import reference_rows
from masslab.presenters.workspace import WorkspacePresenter


class AlloyPresenter:
    """Задание 3: определить сплав по изотопному спектру."""

    def __init__(self, view, rng, np_rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.003,
                                      peak_threshold=0.01, show_mass=True)
        self._task = None
        self._attempts = 0
        view.check_requested.connect(self._on_check)

    def start(self):
        self._task = AlloyTask(self._rng)
        self._attempts = 0
        self._ws.reset()
        self._view.show_reference(reference_rows(with_isotopes=True))
        self._show_sample()
        self._view.show_feedback("", None)

    def _show_sample(self):
        self._view.clear_inputs()
        self._view.set_options([f"{a.name}: {a.composition_text()}" for a in self._task.options])
        self._ws.set_peaks(self._task.peaks())

    def _on_check(self):
        option = self._view.selected_option()
        if option is None:
            self._view.show_feedback("Выберите сплав.", False)
            return
        self._attempts += 1
        if self._task.check(option):
            self._view.show_feedback("Верно! Задание 3 выполнено.", True)
            self._on_completed(self._attempts)
            return
        alloy = self._task.alloy
        self._task.new_sample()
        self._show_sample()
        self._view.show_feedback(
            f"Неверно. Это был сплав «{alloy.name}» ({alloy.composition_text()}). "
            "Выдан новый образец.", False)
