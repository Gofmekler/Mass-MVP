from masslab.model.elements import ELEMENTS
from masslab.model.spectrum import Peak
from masslab.model.tasks import DemoTask
from masslab.presenters.formatting import num, reference_rows
from masslab.presenters.workspace import WorkspacePresenter

DEFAULT_SELECTION = ("H", "N", "Ar")
MAX_IONS = 8


class DemoPresenter:
    """Задание 1: запуск ионов и контрольные вопросы."""

    def __init__(self, view, rng, np_rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.01)
        self._task = None
        self._answers = []
        self._launched = []
        self._attempts = 0
        view.launch_requested.connect(self._on_launch)
        view.answer_selected.connect(self._on_answer)
        view.check_requested.connect(self._on_check)
        view.workspace.voltage_changed.connect(lambda _: self._update_table())

    def start(self):
        self._task = DemoTask(self._rng)
        self._attempts = 0
        self._launched = []
        self._ws.reset()
        elements = sorted(ELEMENTS.values(), key=lambda e: e.mass)
        self._view.set_element_choices(
            [(e.symbol, f"{e.symbol} — {e.name} ({num(e.mass, 3)})") for e in elements])
        self._view.set_selected_elements(DEFAULT_SELECTION)
        self._view.show_reference(reference_rows())
        self._update_table()
        self._show_questions()
        self._view.show_feedback("", None)

    def _show_questions(self):
        self._answers = [None] * len(self._task.questions)
        self._view.show_questions([(q.text, q.options) for q in self._task.questions])

    def _on_launch(self):
        symbols = self._view.selected_elements()
        if not symbols:
            self._view.show_feedback("Выберите хотя бы один элемент для запуска.", False)
            return
        if len(symbols) > MAX_IONS:
            self._view.show_feedback(f"Можно запустить не более {MAX_IONS} ионов одновременно.",
                                     False)
            return
        self._launched = symbols
        self._ws.set_peaks([Peak(ELEMENTS[s].mass, 1.0, s) for s in symbols])
        self._update_table()
        self._view.show_feedback("", None)

    def _update_table(self):
        tof = self._ws.tof
        rows = []
        for s in sorted(self._launched, key=lambda s: ELEMENTS[s].mass):
            m = ELEMENTS[s].mass
            rows.append([s, num(m, 3), num(tof.velocity(m) / 1000, 1),
                         num(tof.flight_time(m) * 1e6, 3)])
        self._view.show_flight_table(rows)

    def _on_answer(self, question_index, option_index):
        self._answers[question_index] = option_index

    def _on_check(self):
        if None in self._answers:
            self._view.show_feedback("Ответьте на все вопросы.", False)
            return
        self._attempts += 1
        results = self._task.check(self._answers)
        if all(results):
            self._view.show_feedback("Все ответы верны. Задание 1 выполнено!", True)
            self._on_completed(self._attempts)
            return
        wrong = results.count(False)
        self._task.new_questions()
        self._show_questions()
        self._view.show_feedback(
            f"Неверных ответов: {wrong} из {len(results)}. Вопросы заменены новыми — "
            "проведите измерения и ответьте ещё раз.", False)
