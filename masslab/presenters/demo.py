from masslab.model.session import STAGE_TITLES, StageReport
from masslab.model.tasks import DEMO_HINTS, DemoTask
from masslab.presenters.formatting import num, reference_rows
from masslab.presenters.hints import HintTracker
from masslab.presenters.launcher import TABLE_HEADERS, IonLauncher
from masslab.presenters.workspace import WorkspacePresenter
from masslab.views.viewmodels import QuestionItem


class DemoPresenter:
    """Задание 1: запуск ионов и контрольные вопросы.

    Верно решённые вопросы фиксируются, неверные заменяются новыми того же типа.
    """

    def __init__(self, view, rng, np_rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.01, adjustable_length=True)
        self._launcher = IonLauncher(view, self._ws)
        self._hints = HintTracker(view.show_hint, DEMO_HINTS)
        self._task = None
        self._answers = []
        self._locked = []
        self._attempts = 0
        self.reveal = False          # режим преподавателя: показывать верные ответы
        view.answer_selected.connect(self._on_answer)
        view.check_requested.connect(self._on_check)

    def start(self):
        self._task = DemoTask(self._rng)
        n = len(self._task.questions)
        self._answers = [None] * n
        self._locked = [False] * n
        self._attempts = 0
        self._ws.reset()
        self._launcher.reset()
        self._view.show_reference(reference_rows())
        self._show_questions()
        self._view.show_feedback("", None)
        self._hints.reset()

    def _show_questions(self):
        self._view.show_questions([
            QuestionItem(q.text, q.options, self._answers[i], self._locked[i],
                         q.correct_index if self.reveal else None)
            for i, q in enumerate(self._task.questions)])

    def _on_answer(self, question_index, option_index):
        if not self._locked[question_index]:
            self._answers[question_index] = option_index

    def _on_check(self):
        if None in self._answers:
            self._view.show_feedback("Ответьте на все вопросы.", False)
            return
        self._attempts += 1
        results = self._task.check(self._answers)
        if all(results):
            self._locked = [True] * len(results)
            self._show_questions()
            self._view.show_feedback("Все ответы верны. Задание 1 выполнено!", True)
            self._hints.reset()
            self._on_completed(self._attempts, self._report())
            return
        wrong = [i for i, ok in enumerate(results) if not ok]
        for i, ok in enumerate(results):
            self._locked[i] = ok
        self._task.replace(wrong)
        for i in wrong:
            self._answers[i] = None
        self._show_questions()
        numbers = ", ".join(str(i + 1) for i in wrong)
        self._view.show_feedback(
            f"Неверно: {numbers}. Верные ответы зафиксированы, а эти вопросы заменены "
            "новыми — проведите измерения и ответьте ещё раз.", False)
        self._hints.failed()

    def _report(self):
        notes = []
        for i, (q, a) in enumerate(zip(self._task.questions, self._answers)):
            notes.append(f"{i + 1}. {q.text}")
            notes.append(f"    Ответ: {q.options[a]}")
        ions = ", ".join(p.label for p in self._launcher.launched) or "—"
        return StageReport(
            STAGE_TITLES["task1"],
            facts=(("Ускоряющее напряжение U", f"{self._ws.voltage} В"),
                   ("Длина дрейфовой трубки L", f"{num(self._ws.length, 2)} м"),
                   ("Последний запуск", ions)),
            table_headers=TABLE_HEADERS,
            table_rows=tuple(tuple(r) for r in self._launcher.table_rows()),
            notes=tuple(notes),
            plot=self._ws.last_spectrum)
