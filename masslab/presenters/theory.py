"""Презентер методички: навигация по разделам и данные для иллюстраций."""
from masslab.model.elements import ELEMENTS
from masslab.model.physics import VOLTAGE_MAX, VOLTAGE_MIN, VOLTAGE_STEP, TOFPhysics
from masslab.model.spectrum import Peak, build_spectrum, find_peaks
from masslab.model.theory import THEORY_SECTIONS
from masslab.presenters.formatting import num
from masslab.presenters.workspace import PALETTE
from masslab.views.viewmodels import (AccelIon, BuildPeak, Curve, Marker, PlotData, SchemeData,
                                      SchemeIon, TheoryVisual)

RESOLUTION_START = 1000          # В: стартуем с плохого разрешения, чтобы было что улучшать
RESOLUTION_RANGE = (202.5, 209.5)
STEPS = ("Входной тест", "Задание 1", "Задание 2", "Задание 3", "Отчёт PDF")


def _ions(*items):
    """items: (подпись, масса, заряд) → SchemeData при стандартных U и L."""
    tof = TOFPhysics()
    ions = tuple(SchemeIon(label, PALETTE[i % len(PALETTE)], tof.flight_time(m, z) * 1e6)
                 for i, (label, m, z) in enumerate(items))
    return SchemeData(ions, int(tof.voltage), tof.length)


class TheoryPresenter:
    def __init__(self, view, np_rng):
        self._view = view
        self._rng = np_rng
        self._index = 0
        self._voltage = RESOLUTION_START
        view.section_selected.connect(self._go)
        view.next_requested.connect(lambda: self._go(self._index + 1))
        view.prev_requested.connect(lambda: self._go(self._index - 1))
        view.resolution_voltage_changed.connect(self._on_voltage)
        view.set_sections([f"{i + 1}. {s.title}" for i, s in enumerate(THEORY_SECTIONS)])
        view.set_resolution_range(VOLTAGE_MIN, VOLTAGE_MAX, VOLTAGE_STEP, self._voltage)

    def open(self):
        self._show()
        self._view.open()

    def _go(self, index):
        if 0 <= index < len(THEORY_SECTIONS):
            self._index = index
            self._show()

    def _show(self):
        s = THEORY_SECTIONS[self._index]
        n = len(THEORY_SECTIONS)
        points = "<ul>" + "".join(f"<li>{p}</li>" for p in s.points) + "</ul>"
        self._view.show_section(self._index, n, s.title, points, s.formula, s.caption,
                                self._visual(s.visual))
        self._view.set_navigation(self._index > 0, self._index < n - 1)
        if s.visual == "resolution":
            self._update_resolution()

    # --- иллюстрации ----------------------------------------------------

    def _visual(self, kind):
        builders = {
            "build": self._build,
            "scheme": lambda: _ions(("H⁺", 1.008, 1), ("N⁺", 14.007, 1), ("Ar⁺", 39.948, 1),
                                    ("Xe⁺", 131.29, 1)),
            "acceleration": self._acceleration,
            "race": lambda: _ions(("He⁺", 4.0026, 1), ("O⁺", 15.999, 1)),
            "calibration": self._calibration,
            "charge": lambda: _ions(("Ne⁺", 20.18, 1), ("Ar²⁺", 39.948, 2), ("Ar⁺", 39.948, 1)),
            "isotopes": self._isotopes,
            "resolution": lambda: None,
            "steps": lambda: STEPS,
        }
        return TheoryVisual(kind, builders[kind]())

    @staticmethod
    def _build():
        tof = TOFPhysics()
        items = (("H⁺", 1.008, 0.5), ("N⁺", 14.007, 1.0), ("Ar⁺", 39.948, 0.7))
        return tuple(BuildPeak(label, PALETTE[i], tof.flight_time(m) * 1e6, share)
                     for i, (label, m, share) in enumerate(items))

    @staticmethod
    def _acceleration():
        light, heavy = ("He⁺", 4.0026), ("Ar⁺", 39.948)
        ratio = (light[1] / heavy[1]) ** 0.5
        return (AccelIon(light[0], PALETTE[0], 1.0), AccelIon(heavy[0], PALETTE[2], ratio))

    def _calibration(self):
        tof = TOFPhysics()
        peaks = [Peak(39.948, 0.8, "Ar⁺"), Peak(14.007, 1.0, "?")]
        t_max = tof.flight_time(39.948) * 1.25
        t, s = build_spectrum(peaks, tof, t_max, self._rng, noise=0.01)
        t_ar, t_x = (tof.flight_time(m) * 1e6 for m in (39.948, 14.007))
        markers = (Marker(t_ar, f"Ar⁺  {num(t_ar, 2)} мкс", PALETTE[1]),
                   Marker(t_x, f"?  {num(t_x, 2)} мкс", "#FF5252"))
        return PlotData("Спектр: калибрант и неизвестный ион", "Время пролёта t, мкс",
                        "Интенсивность", (Curve(tuple(t * 1e6), tuple(s), "#E0E0E0", width=1.0),),
                        markers, (0.0, t_max * 1e6), (0.0, 1.2))

    @staticmethod
    def _isotopes():
        curves = []
        for symbol, color in (("Cu", PALETTE[2]), ("Zn", PALETTE[0])):
            for k, iso in enumerate(ELEMENTS[symbol].isotopes):
                curves.append(Curve((iso.mass, iso.mass), (0.0, iso.abundance * 100), color,
                                    symbol if k == 0 else "", "-", 7.0))
        markers = tuple(Marker(i.mass, str(i.mass_number), "#BDBDBD")
                        for s in ("Cu", "Zn") for i in ELEMENTS[s].isotopes if i.abundance > 0.03)
        return PlotData("Изотопный состав меди и цинка", "m/z", "Содержание, %",
                        tuple(curves), markers, (61.5, 71.0), (0.0, 85.0))

    # --- интерактивное разрешение ----------------------------------------

    def _on_voltage(self, value):
        self._voltage = int(value)
        self._update_resolution()

    def _update_resolution(self):
        tof = TOFPhysics(self._voltage)
        peaks = [Peak(m, a) for m, a in ELEMENTS["Pb"].isotope_peaks()]
        t_max = tof.flight_time(RESOLUTION_RANGE[1]) * 1.01
        t, s = build_spectrum(peaks, tof, t_max, self._rng, noise=0.004)
        mz = tof.mass_from_time(t)          # работает и для массива времён
        window = (mz >= RESOLUTION_RANGE[0]) & (mz <= RESOLUTION_RANGE[1])
        found = [round(tof.mass_from_time(p.time))
                 for p in find_peaks(t, s, tof.sigma_at_time, 0.01)]
        resolved = 206 in found and 208 in found
        plot = PlotData(f"Изотопы свинца при U = {self._voltage} В", "m/z", "Интенсивность",
                        (Curve(tuple(mz[window]), tuple(s[window]), "#E0E0E0", width=1.2),),
                        tuple(Marker(m, str(m), "#81C784") for m in (204, 206, 207, 208)),
                        RESOLUTION_RANGE, (0.0, 1.15))
        verdict = "изотопы разделены ✓" if resolved else "пики сливаются ✗"
        self._view.show_resolution(
            plot, f"U = {self._voltage} В     R ≈ {tof.resolution(207.2):.0f}     {verdict}")
