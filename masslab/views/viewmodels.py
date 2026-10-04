"""Готовые к отображению данные, которые презентеры передают видам."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Curve:
    x: tuple
    y: tuple
    color: str
    label: str = ""
    style: str = "-"
    width: float = 1.5
    alpha: float = 1.0
    animate: bool = True     # участвует ли в анимации «полёта» ионов


@dataclass(frozen=True)
class Marker:
    x: float
    label: str
    color: str


@dataclass(frozen=True)
class PlotData:
    title: str
    x_label: str
    y_label: str
    curves: tuple = ()
    markers: tuple = ()
    x_lim: tuple = None
    y_lim: tuple = None
    y_log: bool = False


@dataclass(frozen=True)
class SchemeIon:
    label: str
    color: str
    time_us: float           # время пролёта дрейфовой трубки


@dataclass(frozen=True)
class SchemeData:
    """Данные для анимированной схемы прибора."""
    ions: tuple = ()
    voltage: int = 0
    length: float = 0.0


@dataclass(frozen=True)
class QuestionItem:
    text: str
    options: tuple
    selected: int = None
    locked: bool = False     # уже решён верно — изменить нельзя


@dataclass(frozen=True)
class ReportRow:
    title: str
    duration: str
    attempts: str


@dataclass(frozen=True)
class ReportData:
    work_title: str
    student: str
    group: str
    finished_at: str
    rows: tuple = ()
    total: str = ""
    verdict: str = ""
    stages: tuple = ()       # StageReport для PDF
