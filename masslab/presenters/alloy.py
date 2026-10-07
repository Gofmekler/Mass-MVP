from masslab.model.session import STAGE_TITLES, StageReport
from masslab.model.tasks import ALLOY_HINTS, AlloyTask
from masslab.presenters.formatting import num, reference_rows
from masslab.presenters.hints import HintTracker
from masslab.presenters.workspace import WorkspacePresenter

REPORT_PEAKS = 12


class AlloyPresenter:
    """Задание 3: определить сплав по изотопному спектру."""

    def __init__(self, view, rng, np_rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.003,
                                      peak_threshold=0.01, show_mass=True)
        self._hints = HintTracker(view.show_hint, ALLOY_HINTS)
        self._task = None
        self._attempts = 0
        self.reveal = False          # режим преподавателя: показывать ответ
        view.check_requested.connect(self._on_check)

    def start(self):
        self._task = AlloyTask(self._rng)
        self._attempts = 0
        self._ws.reset()
        self._view.show_reference(reference_rows(with_isotopes=True))
        self._show_sample()
        self._view.show_feedback("", None)
        self._hints.reset()

    def _show_sample(self):
        task = self._task
        correct = task.options.index(task.alloy) if self.reveal else None
        self._view.set_options([f"{a.name}: {a.composition_text()}" for a in task.options],
                               correct)
        self._ws.set_peaks(task.peaks())
        self._view.show_answer(
            f"Ответ (режим преподавателя): {task.alloy.name} — {task.alloy.composition_text()}."
            if self.reveal else None)

    def _on_check(self):
        option = self._view.selected_option()
        if option is None:
            self._view.show_feedback("Выберите сплав.", False)
            return
        self._attempts += 1
        if self._task.check(option):
            self._view.show_feedback("Верно! Задание 3 выполнено.", True)
            self._hints.reset()
            self._on_completed(self._attempts, self._report())
            return
        alloy = self._task.alloy
        self._task.new_sample()
        self._show_sample()
        self._view.show_feedback(
            f"Неверно. Это был сплав «{alloy.name}» ({alloy.composition_text()}). "
            "Выдан новый образец.", False)
        self._hints.failed()

    def _report(self):
        alloy = self._task.alloy
        peaks = sorted(self._ws.found_peaks(), key=lambda p: -p[1])[:REPORT_PEAKS]
        top = max((h for _, h in peaks), default=1.0)
        rows = tuple((num(mz, 1), num(h / top * 100, 1) + " %")
                     for mz, h in sorted(peaks))
        return StageReport(
            STAGE_TITLES["task3"],
            facts=(("Ускоряющее напряжение U", f"{self._ws.voltage} В"),
                   ("Длина дрейфовой трубки L", f"{num(self._ws.length, 2)} м"),
                   ("Определённый сплав", alloy.name),
                   ("Состав (масс. %)", alloy.composition_text())),
            table_headers=("m/z", "Относительная высота пика"),
            table_rows=rows,
            plot=self._ws.last_spectrum)
