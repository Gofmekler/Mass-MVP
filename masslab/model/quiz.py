"""Вопросы с выбором ответа и входной тест."""
from dataclasses import dataclass

MIN_OPTIONS = 4
MAX_OPTIONS = 6


@dataclass(frozen=True)
class BankQuestion:
    """Вопрос банка: правильный ответ хранится отдельно от неправильных."""
    text: str
    correct: str
    wrong: tuple


@dataclass(frozen=True)
class ChoiceQuestion:
    """Вопрос в том виде, в каком его видит студент: варианты перемешаны."""
    text: str
    options: tuple
    correct_index: int

    @classmethod
    def create(cls, text, correct, wrong, rng):
        options = [correct, *wrong]
        rng.shuffle(options)
        return cls(text, tuple(options), options.index(correct))

    def is_correct(self, index):
        return index == self.correct_index


class Quiz:
    """Входной тест: случайные вопросы из банка, при пересдаче — другие вопросы."""

    def __init__(self, bank, rng, size=10, pass_score=9):
        if len(bank) < size:
            raise ValueError("В банке меньше вопросов, чем нужно для теста")
        self._bank = list(bank)
        self._rng = rng
        self.size = size
        self.pass_score = pass_score
        self.questions = []
        self.answers = []
        self._previous = set()

    def new_attempt(self):
        """Новый набор вопросов; вопросы предыдущей попытки по возможности не повторяются."""
        fresh = [q for q in self._bank if q not in self._previous]
        pool = fresh if len(fresh) >= self.size else self._bank
        picked = self._rng.sample(pool, self.size)
        self._previous = set(picked)
        self.questions = [ChoiceQuestion.create(q.text, q.correct, q.wrong, self._rng)
                          for q in picked]
        self.answers = [None] * self.size

    def answer(self, question_index, option_index):
        self.answers[question_index] = option_index

    def answered_count(self):
        return sum(a is not None for a in self.answers)

    def all_answered(self):
        return self.answered_count() == self.size

    def score(self):
        return sum(q.is_correct(a) for q, a in zip(self.questions, self.answers))

    def passed(self):
        return self.score() >= self.pass_score

    def wrong_questions(self):
        return [q.text for q, a in zip(self.questions, self.answers) if not q.is_correct(a)]
