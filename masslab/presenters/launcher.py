"""Выбор и запуск ионов — общая часть задания 1 и песочницы."""
from masslab.model.elements import ELEMENTS
from masslab.model.spectrum import Peak
from masslab.model.tasks import ion_label
from masslab.presenters.formatting import num

DEFAULT_SELECTION = ("H", "N", "Ar")
DOUBLE_CHARGE_SHARE = 0.15     # доля двухзарядных ионов
TABLE_HEADERS = ("Ион", "m/z", "v, км/с", "t, мкс", "R")


class IonLauncher:
    def __init__(self, view, workspace):
        self._view = view
        self._ws = workspace
        self.launched = []          # [Peak]
        view.launch_requested.connect(self.launch)
        view.double_charge_toggled.connect(self._on_double_charge)
        view.workspace.voltage_changed.connect(lambda _: self.update_table())
        view.workspace.length_changed.connect(lambda _: self.update_table())

    def reset(self):
        self.launched = []
        elements = sorted(ELEMENTS.values(), key=lambda e: e.mass)
        self._view.set_element_choices(
            [(e.symbol, f"{e.symbol} — {e.name} ({num(e.mass, 3)})") for e in elements])
        self._view.set_selected_elements(DEFAULT_SELECTION)
        self._view.set_double_charge(False)
        self.update_table()

    def launch(self):
        symbols = self._view.selected_elements()
        if not symbols:
            self._view.show_launch_feedback("Выберите хотя бы один элемент для запуска.", False)
            return
        peaks = [Peak(ELEMENTS[s].mass, 1.0, ion_label(s)) for s in symbols]
        if self._view.double_charge_enabled():
            peaks += [Peak(ELEMENTS[s].mass, DOUBLE_CHARGE_SHARE, ion_label(s, 2), charge=2)
                      for s in symbols]
        self.launched = peaks
        self._ws.set_peaks(peaks)
        self.update_table()
        self._view.show_launch_feedback("", None)

    def _on_double_charge(self, _enabled):
        if self.launched:
            self.launch()

    def table_rows(self):
        tof = self._ws.tof
        rows = []
        for p in sorted(self.launched, key=lambda p: tof.flight_time(p.mass, p.charge)):
            rows.append([p.label, num(p.mz, 2), num(tof.velocity(p.mass, p.charge) / 1000, 1),
                         num(tof.flight_time(p.mass, p.charge) * 1e6, 3),
                         f"{tof.resolution(p.mass, p.charge):.0f}"])
        return rows

    def update_table(self):
        self._view.show_flight_table(self.table_rows())
