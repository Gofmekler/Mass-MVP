"""Сессия студента: существует только в памяти, никуда не сохраняется."""
import time
from dataclasses import dataclass

STAGES = ("quiz", "task1", "task2", "task3")
STAGE_TITLES = {
    "quiz": "Входной тест",
    "task1": "Задание 1. Запуск ионов",
    "task2": "Задание 2. Неизвестный элемент",
    "task3": "Задание 3. Состав сплава",
}


@dataclass(frozen=True)
class StageReport:
    """Данные этапа для отчёта: факты, таблица и исследованный график."""
    title: str
    facts: tuple = ()          # ((название, значение), ...)
    table_headers: tuple = ()
    table_rows: tuple = ()
    notes: tuple = ()          # строки текста (например, вопросы и ответы)
    plot: object = None        # PlotData


@dataclass
class StageRecord:
    started_at: float = None
    finished_at: float = None
    attempts: int = 0


class LabSession:
    def __init__(self, student, group, clock=time.monotonic, teacher=False):
        self.student = student
        self.group = group
        self.teacher = teacher     # режим преподавателя (с подсказками ответов)
        self._clock = clock
        self.started_at = clock()
        self.finished_at = None
        self.current = None
        self.records = {stage: StageRecord() for stage in STAGES}
        self.reports = {}      # этап → StageReport для PDF-отчёта

    def start_stage(self, stage):
        self.records[stage].started_at = self._clock()
        self.current = stage

    def finish_stage(self, stage, attempts, report=None):
        if report is not None:
            self.reports[stage] = report
        record = self.records[stage]
        record.finished_at = self._clock()
        record.attempts = attempts
        self.current = None
        if all(r.finished_at is not None for r in self.records.values()):
            self.finished_at = record.finished_at

    @property
    def completed(self):
        return self.finished_at is not None

    def stage_duration(self, stage):
        record = self.records[stage]
        if record.started_at is None:
            return 0.0
        end = record.finished_at if record.finished_at is not None else self._clock()
        return end - record.started_at

    def total_duration(self):
        end = self.finished_at if self.finished_at is not None else self._clock()
        return end - self.started_at
