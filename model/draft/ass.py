# tof_mass_spec_gui_singlefile.py
# Интерактивный TOF-симулятор с базой эталонов, калибровкой и GUI (PyQt5)
# Требования: numpy, scipy, matplotlib, PyQt5

import sys
import os
import json
import csv
import math
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("QtAgg")

from PySide6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------------
# Встроенная база эталонов (можно редактировать и сохранять)
# ---------------------------
DEFAULT_DB = {
    "H2": 2.0,
    "He": 4.0,
    "H2O": 18.0,
    "N2": 28.0,
    "CO": 28.0,
    "O2": 32.0,
    "CO2": 44.0,
    "Ar": 40.0
}
DB_FILENAME = "calibrants.json"

# ---------------------------
# Физическая модель TOF (простая)
# t = a * sqrt(m/z) + t0
# ---------------------------

def theoretical_time(mz, a=0.0015, t0=0.0):
    return a * np.sqrt(np.array(mz)) + t0

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_spectrum(mz_list, intensity_list, a=0.0015, t0=0.0,
                      t_min=0.0, t_max=0.02, n_points=3000,
                      detector_sigma=6e-5, noise_level=0.02, seed=None,
                      background_level=0.005):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(t_min, t_max, n_points)
    signal = np.zeros_like(t)
    for mz, inten in zip(mz_list, intensity_list):
        mu = theoretical_time(mz, a=a, t0=t0)
        signal += gaussian(t, amp=float(inten), mu=mu, sigma=detector_sigma)
    max_amp = np.max(signal) if np.max(signal) > 0 else 1.0
    # gaussian noise
    signal += noise_level * max_amp * np.random.normal(0, 1, size=t.shape)
    # low-frequency background
    signal += background_level * max_amp * np.sin(2 * np.pi * t / (t_max + 1e-9))
    signal[signal < 0] = 0.0
    return t, signal

# ---------------------------
# Анализ: поиск пиков, калибровка
# ---------------------------

def detect_peaks(t, signal, prominence=None, distance=None, height=None):
    peaks, props = find_peaks(signal, prominence=prominence, distance=distance, height=height)
    return peaks, props

def calibrate_by_pairs(t_peaks, known_mz):
    # Модель: t = a * sqrt(mz) + t0
    x = np.sqrt(np.array(known_mz))
    y = np.array(t_peaks)
    # линейная подгонка y = a * x + t0
    def linear(x, a, t0): return a * x + t0
    popt, pcov = curve_fit(linear, x, y)
    a, t0 = float(popt[0]), float(popt[1])
    return a, t0, pcov

def time_to_mz(t_vals, a, t0):
    return ((np.array(t_vals) - t0) / a) ** 2

# ---------------------------
# GUI: Matplotlib canvas
# ---------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=4, dpi=110):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

# ---------------------------
# Основное окно приложения
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TOF Mass Spec — интерактивный симулятор (один файл)")
        self.resize(1100, 700)

        # state
        self.db = self._load_db()
        self.sample_mz = [28.0, 32.0, 44.0, 18.0]
        self.sample_int = [1.0, 0.6, 0.8, 0.5]
        self.current_a = 0.0015
        self.current_t0 = 0.0002
        self.t = None
        self.signal = None
        self.peaks_idx = np.array([], dtype=int)
        self.selected_peak_idx = None  # индекс в массиве self.peaks_idx (index into peaks)
        self.peak_label_objs = []
        self.calib_pairs = []  # list of tuples (peak_index_in_peaks, calibrant_name)
        self.config_path = None

        # central widget layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # left: controls
        left = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left, stretch=0)

        # Tabs: Sample, Params, Calibration
        tabs = QtWidgets.QTabWidget()
        left.addWidget(tabs)

        # --- Tab: Sample composition ---
        tab_sample = QtWidgets.QWidget()
        tabs.addTab(tab_sample, "Состав образца")
        s_layout = QtWidgets.QVBoxLayout(tab_sample)

        # Manual table for m/z and intensity
        self.sample_table = QtWidgets.QTableWidget(0, 2)
        self.sample_table.setHorizontalHeaderLabels(["m/z", "Интенсивность"])
        self.sample_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        s_layout.addWidget(self.sample_table)

        tbl_btns = QtWidgets.QHBoxLayout()
        s_layout.addLayout(tbl_btns)
        self.btn_add_row = QtWidgets.QPushButton("Добавить компонент")
        self.btn_remove_row = QtWidgets.QPushButton("Удалить выделенное")
        self.btn_fill_from_db = QtWidgets.QPushButton("Добавить из базы")
        tbl_btns.addWidget(self.btn_add_row)
        tbl_btns.addWidget(self.btn_remove_row)
        tbl_btns.addWidget(self.btn_fill_from_db)

        # list of database items with multi-select for quick add
        s_layout.addWidget(QtWidgets.QLabel("База эталонов (двойной клик — добавить):"))
        self.db_list = QtWidgets.QListWidget()
        self.db_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._refresh_db_list()
        s_layout.addWidget(self.db_list)

        # --- Tab: Parameters ---
        tab_params = QtWidgets.QWidget()
        tabs.addTab(tab_params, "Параметры генерации")
        p_layout = QtWidgets.QFormLayout(tab_params)
        self.edit_a = QtWidgets.QLineEdit(str(self.current_a))
        self.edit_t0 = QtWidgets.QLineEdit(str(self.current_t0))
        self.edit_noise = QtWidgets.QLineEdit("0.02")
        self.edit_sigma = QtWidgets.QLineEdit(str(6e-5))
        self.edit_tmin = QtWidgets.QLineEdit("0.0")
        self.edit_tmax = QtWidgets.QLineEdit("0.02")
        self.edit_npoints = QtWidgets.QLineEdit("3000")
        p_layout.addRow("a (scale):", self.edit_a)
        p_layout.addRow("t0 (смещение):", self.edit_t0)
        p_layout.addRow("Уровень шума (noise_level):", self.edit_noise)
        p_layout.addRow("Detector sigma (s):", self.edit_sigma)
        p_layout.addRow("t_min (s):", self.edit_tmin)
        p_layout.addRow("t_max (s):", self.edit_tmax)
        p_layout.addRow("Points (n):", self.edit_npoints)

        # generate buttons
        gen_layout = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Сгенерировать спектр")
        self.btn_detect = QtWidgets.QPushButton("Найти пики")
        self.btn_clear = QtWidgets.QPushButton("Очистить выбор пиков")
        gen_layout.addWidget(self.btn_generate)
        gen_layout.addWidget(self.btn_detect)
        gen_layout.addWidget(self.btn_clear)
        left.addLayout(gen_layout)

        # --- Tab: Calibration ---
        tab_calib = QtWidgets.QWidget()
        tabs.addTab(tab_calib, "Калибровка")
        c_layout = QtWidgets.QVBoxLayout(tab_calib)

        c_layout.addWidget(QtWidgets.QLabel("Выберите эталон в списке и нажмите 'Привязать пику' (после выбора пика на графике)"))
        self.calib_combobox = QtWidgets.QComboBox()
        self._refresh_calib_combobox()
        c_layout.addWidget(self.calib_combobox)
        calib_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_bind = QtWidgets.QPushButton("Привязать пику -> выбранный эталон")
        self.btn_unbind = QtWidgets.QPushButton("Удалить привязку")
        calib_btn_layout.addWidget(self.btn_bind)
        calib_btn_layout.addWidget(self.btn_unbind)
        c_layout.addLayout(calib_btn_layout)

        self.calib_table = QtWidgets.QTableWidget(0, 3)
        self.calib_table.setHorizontalHeaderLabels(["Peak (t,ms)", "Эталон", "m/z эталона"])
        self.calib_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        c_layout.addWidget(self.calib_table)

        self.btn_calibrate = QtWidgets.QPushButton("Выполнить калибровку (по привязкам)")
        self.btn_apply_calib = QtWidgets.QPushButton("Применить калибровку к графику")
        c_layout.addWidget(self.btn_calibrate)
        c_layout.addWidget(self.btn_apply_calib)

        # Buttons for DB edit and save/load
        db_ops = QtWidgets.QHBoxLayout()
        self.btn_db_add = QtWidgets.QPushButton("Добавить эталон")
        self.btn_db_edit = QtWidgets.QPushButton("Изменить эталон")
        self.btn_db_remove = QtWidgets.QPushButton("Удалить эталон")
        self.btn_db_save = QtWidgets.QPushButton("Сохранить базу")
        db_ops.addWidget(self.btn_db_add)
        db_ops.addWidget(self.btn_db_edit)
        db_ops.addWidget(self.btn_db_remove)
        db_ops.addWidget(self.btn_db_save)
        left.addLayout(db_ops)

        # Save/load experiment / export
        file_ops = QtWidgets.QHBoxLayout()
        self.btn_save_config = QtWidgets.QPushButton("Сохранить эксперимент")
        self.btn_load_config = QtWidgets.QPushButton("Загрузить эксперимент")
        self.btn_export_csv = QtWidgets.QPushButton("Экспорт CSV")
        self.btn_export_report = QtWidgets.QPushButton("Экспорт отчёта")
        file_ops.addWidget(self.btn_save_config)
        file_ops.addWidget(self.btn_load_config)
        file_ops.addWidget(self.btn_export_csv)
        file_ops.addWidget(self.btn_export_report)
        left.addLayout(file_ops)

        # Right: plot
        right = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right, stretch=1)
        self.canvas = MplCanvas(self, width=8, height=5)
        right.addWidget(self.canvas)

        # status area
        self.status = QtWidgets.QLabel("Готово")
        right.addWidget(self.status)

        # connect signals
        self.btn_add_row.clicked.connect(self.on_add_row)
        self.btn_remove_row.clicked.connect(self.on_remove_row)
        self.btn_fill_from_db.clicked.connect(self.on_fill_from_db)
        self.db_list.itemDoubleClicked.connect(self.on_db_item_double)
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_detect.clicked.connect(self.on_detect)
        self.btn_clear.clicked.connect(self.on_clear_selection)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.btn_bind.clicked.connect(self.on_bind_peak)
        self.btn_unbind.clicked.connect(self.on_unbind_peak)
        self.btn_calibrate.clicked.connect(self.on_calibrate)
        self.btn_apply_calib.clicked.connect(self.on_apply_calib)
        self.btn_db_add.clicked.connect(self.on_db_add)
        self.btn_db_edit.clicked.connect(self.on_db_edit)
        self.btn_db_remove.clicked.connect(self.on_db_remove)
        self.btn_db_save.clicked.connect(self.on_db_save)
        self.btn_save_config.clicked.connect(self.on_save_config)
        self.btn_load_config.clicked.connect(self.on_load_config)
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_export_report.clicked.connect(self.on_export_report)

        # fill sample table initial
        self._populate_sample_table()
        # initial generate
        self.on_generate()

    # ---------------------------
    # DB helpers
    # ---------------------------
    def _load_db(self):
        if os.path.exists(DB_FILENAME):
            try:
                with open(DB_FILENAME, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # validate: values numeric
                    good = {k: float(v) for k, v in data.items()}
                    return good
            except Exception:
                pass
        # fallback to default
        return dict(DEFAULT_DB)

    def _refresh_db_list(self):
        self.db_list.clear()
        for name, mz in sorted(self.db.items()):
            self.db_list.addItem(f"{name}\t{mz}")

    def _refresh_calib_combobox(self):
        self.calib_combobox.clear()
        for name in sorted(self.db.keys()):
            self.calib_combobox.addItem(name)

    # ---------------------------
    # Sample table helpers
    # ---------------------------
    def _populate_sample_table(self):
        self.sample_table.setRowCount(0)
        for mz, inten in zip(self.sample_mz, self.sample_int):
            r = self.sample_table.rowCount()
            self.sample_table.insertRow(r)
            mz_item = QtWidgets.QTableWidgetItem(str(mz))
            inten_item = QtWidgets.QTableWidgetItem(str(inten))
            self.sample_table.setItem(r, 0, mz_item)
            self.sample_table.setItem(r, 1, inten_item)

    def on_add_row(self):
        r = self.sample_table.rowCount()
        self.sample_table.insertRow(r)
        mz_item = QtWidgets.QTableWidgetItem("100.0")
        inten_item = QtWidgets.QTableWidgetItem("0.5")
        self.sample_table.setItem(r, 0, mz_item)
        self.sample_table.setItem(r, 1, inten_item)

    def on_remove_row(self):
        rows = sorted({idx.row() for idx in self.sample_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.sample_table.removeRow(r)

    def on_fill_from_db(self):
        items = self.db_list.selectedItems()
        if not items:
            QtWidgets.QMessageBox.information(self, "Добавление из базы", "Выберите элементы в базе (выделите) и нажмите OK.")
            return
        for it in items:
            text = it.text()
            # format is "name\tmz"
            if "\t" in text:
                name, mz = text.split("\t", 1)
            else:
                parts = text.split()
                name = parts[0]
                mz = parts[-1]
            r = self.sample_table.rowCount()
            self.sample_table.insertRow(r)
            self.sample_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(mz)))
            self.sample_table.setItem(r, 1, QtWidgets.QTableWidgetItem("1.0"))

    def on_db_item_double(self, item):
        # add to sample table
        text = item.text()
        if "\t" in text:
            name, mz = text.split("\t", 1)
        else:
            parts = text.split()
            name = parts[0]
            mz = parts[-1]
        r = self.sample_table.rowCount()
        self.sample_table.insertRow(r)
        self.sample_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(mz)))
        self.sample_table.setItem(r, 1, QtWidgets.QTableWidgetItem("1.0"))

    # ---------------------------
    # Generate & detect
    # ---------------------------
    def _read_sample_from_table(self):
        mzs = []
        ints = []
        for r in range(self.sample_table.rowCount()):
            try:
                mz = float(self.sample_table.item(r,0).text())
                inten = float(self.sample_table.item(r,1).text())
                mzs.append(mz)
                ints.append(inten)
            except Exception:
                continue
        if not mzs:
            raise ValueError("Состав образца пуст или неверно введён.")
        return mzs, ints

    def on_generate(self):
        try:
            mzs, ints = self._read_sample_from_table()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))
            return
        # read params
        try:
            a = float(self.edit_a.text())
            t0 = float(self.edit_t0.text())
            noise = float(self.edit_noise.text())
            sigma = float(self.edit_sigma.text())
            tmin = float(self.edit_tmin.text())
            tmax = float(self.edit_tmax.text())
            npoints = int(float(self.edit_npoints.text()))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка параметров", "Проверь параметры генерации.")
            return
        self.current_a = a
        self.current_t0 = t0
        self.t, self.signal = generate_spectrum(mzs, ints, a=a, t0=t0,
                                               t_min=tmin, t_max=tmax, n_points=npoints,
                                               detector_sigma=sigma, noise_level=noise)
        self.peaks_idx = np.array([], dtype=int)
        self.selected_peak_idx = None
        self.calib_pairs = []
        self.calib_table.setRowCount(0)
        self.plot_signal()
        self.status.setText("Спектр сгенерирован.")

    def on_detect(self):
        if self.t is None:
            QtWidgets.QMessageBox.information(self, "Поиск пиков", "Сначала сгенерируйте спектр.")
            return
        prom = 0.05 * np.max(self.signal)
        distance = max(5, int(len(self.t)/500))  # adaptive
        peaks, props = detect_peaks(self.t, self.signal, prominence=prom, distance=distance)
        self.peaks_idx = peaks
        self.selected_peak_idx = None
        self.plot_signal()
        self.status.setText(f"Найдено пиков: {len(peaks)}")

    def on_clear_selection(self):
        self.selected_peak_idx = None
        self.calib_pairs = []
        self.calib_table.setRowCount(0)
        self.plot_signal()

    # ---------------------------
    # Plot interactions
    # ---------------------------
    def plot_signal(self):
        ax = self.canvas.ax
        ax.clear()
        if self.t is None:
            self.canvas.draw()
            return
        ax.plot(self.t*1e3, self.signal, label='Сигнал')
        ax.set_xlabel("Время, ms")
        ax.set_ylabel("Интенсивность (отн.)")
        # peaks
        if len(self.peaks_idx)>0:
            ax.plot(self.t[self.peaks_idx]*1e3, self.signal[self.peaks_idx], 'x', label='Пики')
            for i, idx in enumerate(self.peaks_idx):
                tx = self.t[idx]*1e3
                ty = self.signal[idx]
                label = f"{tx:.3f} ms"
                # if calibrated, show m/z near peaks (if this peak is not selected for binding, show both)
                # find if this peak has a binding
                bound = None
                for pidx, name in self.calib_pairs:
                    if pidx == idx:
                        bound = name
                        break
                if bound and self.current_a is not None:
                    mz = time_to_mz(self.t[idx], self.current_a, self.current_t0)
                    label = f"{bound}\n{mz:.2f}"
                ann = ax.annotate(label, (tx, ty), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
                self.peak_label_objs.append(ann)
        # highlight selected peak
        if self.selected_peak_idx is not None and self.selected_peak_idx < len(self.peaks_idx):
            idx = self.peaks_idx[self.selected_peak_idx]
            ax.plot(self.t[idx]*1e3, self.signal[idx], 'o', markersize=10, fillstyle='none', markeredgewidth=2)
        ax.legend()
        self.canvas.draw()

    def on_plot_click(self, event):
        # only left click on axes
        if event.inaxes is None:
            return
        if self.t is None or len(self.peaks_idx)==0:
            return
        x_ms = event.xdata  # ms
        # convert to seconds and find nearest peak
        x_s = x_ms / 1000.0
        distances = np.abs(self.t[self.peaks_idx] - x_s)
        nearest_idx = int(np.argmin(distances))
        if distances[nearest_idx] > ( (self.t[1]-self.t[0]) * 10 ):  # too far
            return
        self.selected_peak_idx = nearest_idx
        sel_peak_global_idx = int(self.peaks_idx[self.selected_peak_idx])
        self.status.setText(f"Выбран пик: t = {self.t[sel_peak_global_idx]*1e3:.4f} ms")
        self.plot_signal()

    # ---------------------------
    # Calibration: bind peaks to calibrants
    # ---------------------------
    def on_bind_peak(self):
        if self.selected_peak_idx is None:
            QtWidgets.QMessageBox.information(self, "Привязка", "Сначала выберите пик на графике (щелчок).")
            return
        name = self.calib_combobox.currentText()
        if not name:
            return
        global_idx = int(self.peaks_idx[self.selected_peak_idx])
        # check if already bound
        for pidx, nm in self.calib_pairs:
            if pidx == global_idx:
                QtWidgets.QMessageBox.information(self, "Привязка", "Этот пик уже привязан.")
                return
        self.calib_pairs.append((global_idx, name))
        self._refresh_calib_table()
        self.plot_signal()

    def on_unbind_peak(self):
        # remove selected row in calib_table
        sel = self.calib_table.selectedIndexes()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Удалить привязку", "Выберите строку в таблице привязок.")
            return
        row = sel[0].row()
        # remove from list
        if 0 <= row < len(self.calib_pairs):
            self.calib_pairs.pop(row)
            self._refresh_calib_table()
            self.plot_signal()

    def _refresh_calib_table(self):
        self.calib_table.setRowCount(0)
        for pidx, name in self.calib_pairs:
            r = self.calib_table.rowCount()
            self.calib_table.insertRow(r)
            t_val = self.t[pidx]*1e3
            mz_val = self.db.get(name, 0.0)
            self.calib_table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{t_val:.4f}"))
            self.calib_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(name)))
            self.calib_table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(mz_val)))

    def on_calibrate(self):
        if len(self.calib_pairs) < 2:
            QtWidgets.QMessageBox.information(self, "Калибровка", "Нужно минимум 2 привязки для калибровки.")
            return
        t_peaks = []
        known_mz = []
        for pidx, name in self.calib_pairs:
            t_peaks.append(self.t[pidx])
            known_mz.append(self.db.get(name, None))
        try:
            a, t0, pcov = calibrate_by_pairs(t_peaks, known_mz)
            self.current_a = a
            self.current_t0 = t0
            self.edit_a.setText(str(a))
            self.edit_t0.setText(str(t0))
            QtWidgets.QMessageBox.information(self, "Калибровка", f"Готово: a={a:.6e}, t0={t0:.6e}")
            self.status.setText("Калибровка выполнена.")
            self.plot_signal()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка калибровки", str(e))

    def on_apply_calib(self):
        # apply to plot (just replot will show m/z labels for bound peaks)
        try:
            self.current_a = float(self.edit_a.text())
            self.current_t0 = float(self.edit_t0.text())
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Неверные параметры a/t0")
            return
        self.plot_signal()
        self.status.setText("Калибровка применена к графику.")

    # ---------------------------
    # DB edit
    # ---------------------------
    def on_db_add(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Добавить эталон", "Название (пример: CO2):")
        if not ok or not name.strip():
            return
        mz_text, ok2 = QtWidgets.QInputDialog.getText(self, "Добавить эталон", "m/z (число):")
        if not ok2:
            return
        try:
            mz = float(mz_text)
        except:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "m/z должен быть числом.")
            return
        self.db[name.strip()] = float(mz)
        self._refresh_db_list()
        self._refresh_calib_combobox()

    def on_db_edit(self):
        current = self.calib_combobox.currentText()
        if not current:
            QtWidgets.QMessageBox.information(self, "Редактирование", "Выберите эталон в выпадающем списке.")
            return
        mz_text, ok = QtWidgets.QInputDialog.getText(self, "Изменить m/z", f"Новое m/z для {current}:", text=str(self.db.get(current, "")))
        if not ok:
            return
        try:
            mz = float(mz_text)
        except:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "m/z должен быть числом.")
            return
        self.db[current] = float(mz)
        self._refresh_db_list()
        self._refresh_calib_combobox()
        self._refresh_calib_table()

    def on_db_remove(self):
        current = self.calib_combobox.currentText()
        if not current:
            QtWidgets.QMessageBox.information(self, "Удалить эталон", "Выберите эталон.")
            return
        reply = QtWidgets.QMessageBox.question(self, "Подтвердите удаление", f"Удалить эталон {current}?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            if current in self.db:
                del self.db[current]
            self._refresh_db_list()
            self._refresh_calib_combobox()
            self._refresh_calib_table()

    def on_db_save(self):
        try:
            with open(DB_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(self.db, f, indent=2, ensure_ascii=False)
            QtWidgets.QMessageBox.information(self, "Сохранено", f"База сохранена в {DB_FILENAME}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка сохранения", str(e))

    # ---------------------------
    # Save/load experiment
    # ---------------------------
    def on_save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить эксперимент", filter="JSON files (*.json)")
        if not path:
            return
        try:
            mzs, ints = self._read_sample_from_table()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", str(e))
            return
        data = {
            "sample": {"mzs": mzs, "ints": ints},
            "params": {
                "a": float(self.edit_a.text()),
                "t0": float(self.edit_t0.text()),
                "noise": float(self.edit_noise.text()),
                "sigma": float(self.edit_sigma.text()),
                "tmin": float(self.edit_tmin.text()),
                "tmax": float(self.edit_tmax.text()),
                "npoints": int(float(self.edit_npoints.text()))
            },
            "db": self.db,
            "calib_pairs": [(int(pidx), name) for (pidx, name) in self.calib_pairs]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        QtWidgets.QMessageBox.information(self, "Сохранено", f"Эксперимент сохранён в {path}")

    def on_load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Загрузить эксперимент", filter="JSON files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sample = data.get("sample", {})
            mzs = sample.get("mzs", [])
            ints = sample.get("ints", [])
            self.sample_table.setRowCount(0)
            for mz, inten in zip(mzs, ints):
                r = self.sample_table.rowCount()
                self.sample_table.insertRow(r)
                self.sample_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(mz)))
                self.sample_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(inten)))
            params = data.get("params", {})
            self.edit_a.setText(str(params.get("a", self.edit_a.text())))
            self.edit_t0.setText(str(params.get("t0", self.edit_t0.text())))
            self.edit_noise.setText(str(params.get("noise", self.edit_noise.text())))
            self.edit_sigma.setText(str(params.get("sigma", self.edit_sigma.text())))
            self.edit_tmin.setText(str(params.get("tmin", self.edit_tmin.text())))
            self.edit_tmax.setText(str(params.get("tmax", self.edit_tmax.text())))
            self.edit_npoints.setText(str(params.get("npoints", self.edit_npoints.text())))

            # load db if present
            db = data.get("db", None)
            if db:
                self.db = {k: float(v) for k, v in db.items()}
            self._refresh_db_list()
            self._refresh_calib_combobox()

            # calib pairs
            cp = data.get("calib_pairs", [])
            self.calib_pairs = [(int(pidx), name) for (pidx, name) in cp]
            self._refresh_calib_table()

            self.on_generate()
            QtWidgets.QMessageBox.information(self, "Загружено", f"Эксперимент загружен из {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка загрузки", str(e))

    # ---------------------------
    # Export CSV / Report
    # ---------------------------
    def on_export_csv(self):
        if self.t is None:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет данных для экспорта.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт CSV", filter="CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time_s', 'signal'])
                for ti, si in zip(self.t, self.signal):
                    w.writerow([f"{ti:.9e}", f"{si:.9e}"])
            QtWidgets.QMessageBox.information(self, "Готово", f"CSV сохранён: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))

    def on_export_report(self):
        if self.t is None:
            QtWidgets.QMessageBox.information(self, "Экспорт", "Нет данных для отчёта.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Экспорт отчёта", filter="Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("Отчёт — симуляция TOF масс-спектрометра\n")
                f.write("=====================================\n\n")
                f.write(f"Калибровка: a = {self.current_a:.6e}, t0 = {self.current_t0:.6e}\n\n")
                f.write("Параметры генерации:\n")
                f.write(f"noise = {self.edit_noise.text()}, sigma = {self.edit_sigma.text()}\n\n")
                if len(self.peaks_idx)>0:
                    f.write("Найденные пики:\n")
                    f.write("№\t t_peak(s)\t intensity\t m/z (по калибровке)\n")
                    for i, idx in enumerate(self.peaks_idx, start=1):
                        tpk = self.t[idx]
                        sig = self.signal[idx]
                        mz = time_to_mz(tpk, self.current_a, self.current_t0)
                        f.write(f"{i}\t{tpk:.6e}\t{sig:.6e}\t{mz:.4f}\n")
                else:
                    f.write("Пики не найдены.\n")
                f.write("\nПривязки:\n")
                for pidx, name in self.calib_pairs:
                    f.write(f"t={self.t[pidx]*1e3:.4f} ms <-> {name} (m/z={self.db.get(name)})\n")
            QtWidgets.QMessageBox.information(self, "Готово", f"Отчёт экспортирован в {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
