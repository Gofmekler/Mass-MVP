from masslab.model.elements import ELEMENTS
from masslab.model.session import STAGE_TITLES, StageReport
from masslab.model.tasks import (CALIBRANTS, ELEMENT_HINTS, MASS_TOLERANCE, TIME_TOLERANCE,
                                 ElementTask)
from masslab.presenters.formatting import num, parse_number, reference_rows
from masslab.presenters.hints import HintTracker
from masslab.presenters.workspace import WorkspacePresenter


class ElementPresenter:
    """Задание 2: измерить время, вычислить массу и определить элемент."""

    def __init__(self, view, rng, np_rng, on_completed):
        self._view = view
        self._rng = rng
        self._on_completed = on_completed
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.02)
        self._hints = HintTracker(view.show_hint, ELEMENT_HINTS)
        self._task = None
        self._attempts = 0
        view.check_requested.connect(self._on_check)

    def start(self):
        self._task = ElementTask(self._rng)
        self._attempts = 0
        self._ws.reset()
        self._view.show_reference(reference_rows())
        self._show_sample()
        self._view.show_feedback("", None)
        self._hints.reset()

    def _show_sample(self):
        self._view.clear_inputs()
        self._view.set_options([f"{s} — {ELEMENTS[s].name}" for s in self._task.options])
        self._ws.set_peaks(self._task.peaks())

    def _on_check(self):
        t_us = parse_number(self._view.time_text())
        if t_us is None:
            self._view.show_feedback("Введите время пролёта в микросекундах, например 5,90.",
                                     False)
            return
        mass = parse_number(self._view.mass_text())
        if mass is None:
            self._view.show_feedback("Введите массу числом, например 22,9.", False)
            return
        option = self._view.selected_option()
        if option is None:
            self._view.show_feedback("Выберите элемент.", False)
            return
        self._attempts += 1
        tof = self._ws.tof
        result = self._task.check(tof, t_us * 1e-6, mass, option)
        if result.passed:
            self._view.show_feedback("Верно! Задание 2 выполнено.", True)
            self._hints.reset()
            self._on_completed(self._attempts, self._report(t_us, mass))
            return
        e = self._task.unknown
        true_t = tof.flight_time(e.mass) * 1e6
        self._task.new_sample()
        self._show_sample()
        self._view.show_feedback(
            f"{self._diagnosis(result)}\nПравильно: t = {num(true_t, 3)} мкс, "
            f"m = {num(e.mass, 2)} а.е.м., {e.symbol} — {e.name}. Выдан новый образец.", False)
        self._hints.failed()

    @staticmethod
    def _diagnosis(r):
        if not r.time_ok and r.calculation_ok:
            return (f"Расчёт по вашему времени выполнен верно, но время пролёта измерено "
                    f"неточно (допуск ±{num(TIME_TOLERANCE * 100, 1)} %). Измеряйте время "
                    f"при текущем напряжении.")
        if not r.time_ok:
            return "Неверно измерено время пролёта, и масса не соответствует введённому времени."
        if not r.mass_ok:
            return (f"Время измерено верно, но масса вычислена с ошибкой (допуск "
                    f"±{MASS_TOLERANCE * 100:.0f} %). Проверьте формулу и перевод мкс в секунды.")
        return "Время и масса определены верно, но элемент выбран неверно — сверьтесь со справочником."

    def _report(self, t_us, mass):
        e = self._task.unknown
        error = abs(mass - e.mass) / e.mass * 100
        return StageReport(
            STAGE_TITLES["task2"],
            facts=(("Ускоряющее напряжение U", f"{self._ws.voltage} В"),
                   ("Длина дрейфовой трубки L", f"{num(self._ws.length, 2)} м"),
                   ("Калибровочная смесь", ", ".join(CALIBRANTS)),
                   ("Измеренное время пролёта", f"{num(t_us, 3)} мкс"),
                   ("Вычисленная масса", f"{num(mass, 2)} а.е.м."),
                   ("Определённый элемент", f"{e.symbol} — {e.name}"),
                   ("Табличная масса", f"{num(e.mass, 3)} а.е.м."),
                   ("Погрешность определения массы", f"{num(error, 2)} %")),
            plot=self._ws.last_spectrum)
