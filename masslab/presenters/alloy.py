from masslab.model.elements import ELEMENTS
from masslab.model.session import STAGE_TITLES, StageReport
from masslab.model.tasks import ALLOY_HINTS, AlloyTask
from masslab.presenters.formatting import num, reference_rows
from masslab.presenters.hints import HintTracker
from masslab.presenters.workspace import WorkspacePresenter

REPORT_PEAKS = 12


def _names(symbols):
    return ", ".join(symbols)


class AlloyPresenter:
    """Задание 3 в два шага: 1) какие элементы есть в образце, 2) какой это сплав.

    Ошибка на шаге 1 не меняет образец — отметки можно исправить. Ошибка на шаге 2
    выдаёт новый образец, и работа начинается с шага 1.
    """

    def __init__(self, view, rng, np_rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.003,
                                      peak_threshold=0.01, show_mass=True)
        self._hints = HintTracker(view.show_hint, ALLOY_HINTS)
        self._task = None
        self._attempts = 0
        self._step = 1
        self.reveal = False          # режим преподавателя: показывать ответ
        view.elements_check_requested.connect(self._on_elements_check)
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
        self._step = 1
        alloy = task.alloy
        self._view.set_element_choices(
            [(s, f"{s} — {ELEMENTS[s].name}") for s in task.element_choices()],
            alloy.major_symbols if self.reveal else None)
        correct = task.options.index(alloy) if self.reveal else None
        self._view.set_options([f"{a.name}: {a.composition_text()}" for a in task.options],
                               correct)
        self._view.set_step(1)
        self._view.show_elements_feedback("", None)
        self._ws.set_peaks(task.peaks())
        self._view.show_answer(
            f"Ответ (режим преподавателя): элементы {_names(alloy.major_symbols)}; "
            f"сплав {alloy.name} — {alloy.composition_text()}." if self.reveal else None)

    # --- шаг 1: элементы ------------------------------------------------

    def _on_elements_check(self):
        if self._step != 1:
            return
        selected = self._view.selected_elements()
        if not selected:
            self._view.show_elements_feedback("Отметьте элементы, пики которых видны в спектре.",
                                              False)
            return
        self._attempts += 1
        result = self._task.check_elements(selected)
        alloy = self._task.alloy
        if result.passed:
            self._step = 2
            self._view.set_step(2)
            text = f"Верно: в образце есть {_names(alloy.major_symbols)}."
            if alloy.minor_symbols:
                text += f" Есть и малые добавки: {_names(alloy.minor_symbols)}."
            self._view.show_elements_feedback(text + " Перейдите к шагу 2.", True)
            self._hints.reset()
            return
        parts = []
        if result.missing:
            parts.append(f"не отмечено основных элементов: {len(result.missing)}")
        if result.extra:
            parts.append(f"отмечено лишних: {len(result.extra)}")
        self._view.show_elements_feedback(
            "Пока неверно — " + ", ".join(parts) + ". Сравните пики со справочником "
            "изотопов и исправьте отметки.", False)
        self._hints.failed()

    # --- шаг 2: сплав ---------------------------------------------------

    def _on_check(self):
        if self._step != 2:
            return
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
            "Выдан новый образец — начните с шага 1.", False)
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
                   ("Найденные элементы", _names(alloy.major_symbols)),
                   ("Определённый сплав", alloy.name),
                   ("Состав (масс. %)", alloy.composition_text())),
            table_headers=("m/z", "Относительная высота пика"),
            table_rows=rows,
            plot=self._ws.last_spectrum)
