from masslab.model.defense import DefenseTask
from masslab.model.elements import ELEMENTS
from masslab.model.session import STAGE_TITLES, StageReport
from masslab.presenters.formatting import num
from masslab.presenters.hints import HintTracker
from masslab.views.viewmodels import QuestionItem

DEFENSE_HINTS = (
    "Время пролёта t = L·√(m / 2qU): время растёт с массой и длиной трубки и убывает "
    "с ростом напряжения и заряда.",
    "Если величина под корнем меняется в k раз, время меняется в √k раз. "
    "Двухзарядный ион ведёт себя как однозарядный с массой m/2.",
    "Несколько пиков одного элемента — это его изотопы. Чем тяжелее элемент, тем "
    "дальше по времени его пики.",
)


class DefensePresenter:
    """Защита: вопросы по результатам заданий 2 и 3 (как контрольные вопросы задания 1)."""

    def __init__(self, view, rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._hints = HintTracker(view.show_hint, DEFENSE_HINTS)
        self._task = None
        self._answers = []
        self._locked = []
        self._attempts = 0
        self.reveal = False          # режим преподавателя: показывать верные ответы
        self.element = None          # ElementResult задания 2
        self.alloy = None            # Alloy задания 3
        view.answer_selected.connect(self._on_answer)
        view.check_requested.connect(self._on_check)

    def start(self):
        self._task = DefenseTask(self._rng, self.element, self.alloy)
        n = len(self._task.questions)
        self._answers = [None] * n
        self._locked = [False] * n
        self._attempts = 0
        e = self.element
        self._view.show_summary(
            f"Ваши результаты. Задание 2: {e.symbol} — {ELEMENTS[e.symbol].name}, "
            f"t = {num(e.time_us, 3)} мкс при U = {e.voltage} В, L = {num(e.length, 2)} м. "
            f"Задание 3: сплав {self.alloy.name} ({self.alloy.composition_text()}).")
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
            self._view.show_feedback("Все ответы верны. Работа защищена!", True)
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
            "новыми.", False)
        self._hints.failed()

    def _report(self):
        notes = []
        for i, (q, a) in enumerate(zip(self._task.questions, self._answers)):
            notes.append(f"{i + 1}. {q.text}")
            notes.append(f"    Ответ: {q.options[a]}")
        return StageReport(STAGE_TITLES["defense"], notes=tuple(notes))
