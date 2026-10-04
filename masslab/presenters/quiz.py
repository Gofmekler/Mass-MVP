from masslab.model.question_bank import QUESTION_BANK
from masslab.model.quiz import Quiz


class QuizPresenter:
    """Входной тест: 10 вопросов, зачёт от 9 правильных, пересдача — с новыми вопросами."""

    def __init__(self, view, rng, on_completed, bank=QUESTION_BANK):
        self._view = view
        self._rng = rng
        self._bank = bank
        self._on_completed = on_completed
        self._quiz = None
        self._index = 0
        self._attempts = 0
        self._passed = False
        view.answer_selected.connect(self._on_answer)
        view.next_requested.connect(lambda: self._go(self._index + 1))
        view.prev_requested.connect(lambda: self._go(self._index - 1))
        view.finish_requested.connect(self._on_finish)
        view.result_action_requested.connect(self._on_result_action)

    def start(self):
        self._quiz = Quiz(self._bank, self._rng)
        self._attempts = 0
        self._new_attempt()

    def _new_attempt(self):
        self._quiz.new_attempt()
        self._attempts += 1
        self._passed = False
        self._index = 0
        self._show()

    def _show(self):
        q = self._quiz.questions[self._index]
        self._view.show_question(self._index + 1, self._quiz.size, q.text, q.options,
                                 self._quiz.answers[self._index])
        self._update_controls()

    def _update_controls(self):
        n = self._quiz.size
        self._view.set_navigation(self._index > 0, self._index < n - 1,
                                  self._quiz.all_answered())
        self._view.set_progress(self._quiz.answered_count(), n)

    def _go(self, index):
        if 0 <= index < self._quiz.size:
            self._index = index
            self._show()

    def _on_answer(self, option_index):
        self._quiz.answer(self._index, option_index)
        self._update_controls()

    def _on_finish(self):
        if not self._quiz.all_answered():
            return
        quiz = self._quiz
        score = quiz.score()
        self._passed = quiz.passed()
        if self._passed:
            self._view.show_result(
                "Тест сдан",
                f"Правильных ответов: {score} из {quiz.size}. Можно приступать к выполнению работы.",
                [], True, "Перейти к заданиям")
        else:
            self._view.show_result(
                "Тест не сдан",
                f"Правильных ответов: {score} из {quiz.size}, нужно не менее {quiz.pass_score}. "
                "Повторите теорию и пройдите тест ещё раз — вопросы будут другими.",
                ["Вопросы, на которые дан неверный ответ:"] + quiz.wrong_questions(),
                False, "Пройти тест заново")

    def _on_result_action(self):
        if self._passed:
            self._on_completed(self._attempts)
        else:
            self._new_attempt()
