from PySide6 import QtWidgets, QtCore, QtGui
from view.mpl_panel import MplPanel
import numpy as np


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.ctrl = controller
        self.setWindowTitle("MassLab — TOF Mass Spectrometer")

        # Для всплывающих подсказок
        self.traj_annotation = None
        self.spec_annotation = None
        self.comparison_visible = False

        # Для хранения данных о пиках
        self.peak_annotations = []

        # Список референсных элементов
        self.reference_elements = ["H", "He", "C", "N", "O", "Na", "Mg", "Al", "Si", "Fe", "Cu"]

        self._init_ui()
        self.update_plots()

    def _init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Простой темный стиль
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-size: 11px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 8px;
                background-color: #2A2A2A;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 5px 0 5px;
                color: #4CAF50;
            }
            QLabel {
                color: #E0E0E0;
            }
            QLineEdit {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
            QComboBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
                min-height: 25px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                color: white;
                selection-background-color: #4CAF50;
            }
            QSlider::groove:horizontal {
                background: #444;
                height: 5px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QPushButton {
                background-color: #3A3A3A;
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QCheckBox {
                color: #E0E0E0;
                font-size: 10px;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2A2A2A;
            }
            QTabBar::tab {
                background-color: #333;
                color: #AAA;
                padding: 5px 10px;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background-color: #2A2A2A;
                color: white;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #2A2A2A;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background-color: #4CAF50;
                border-radius: 4px;
                min-height: 20px;
            }
        """)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # === ЛЕВАЯ ПАНЕЛЬ С ПРОКРУТКОЙ ===
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(3)

        # 1. РЕЖИМ РАБОТЫ
        mode_box = QtWidgets.QGroupBox("Режим работы")
        mode_layout = QtWidgets.QHBoxLayout(mode_box)

        self.btn_single = QtWidgets.QPushButton("Одиночный элемент")
        self.btn_mixture = QtWidgets.QPushButton("Смесь газов")

        self.btn_single.setCheckable(True)
        self.btn_mixture.setCheckable(True)
        self.btn_single.setChecked(True)

        self.btn_single.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:checked {
                background-color: #388E3C;
            }
        """)

        self.btn_mixture.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
            }
            QPushButton:checked {
                background-color: #1976D2;
            }
        """)

        self.btn_single.clicked.connect(lambda: self.set_mode("single"))
        self.btn_mixture.clicked.connect(lambda: self.set_mode("mixture"))

        mode_layout.addWidget(self.btn_single)
        mode_layout.addWidget(self.btn_mixture)
        left_layout.addWidget(mode_box)

        # 2. НАПРЯЖЕНИЕ
        voltage_box = QtWidgets.QGroupBox("Напряжение")
        voltage_layout = QtWidgets.QVBoxLayout(voltage_box)

        self.lbl_U = QtWidgets.QLabel(f"U = {self.ctrl.get_voltage():.0f} V")
        self.lbl_U.setStyleSheet("font-weight: bold; color: #4CAF50;")

        self.slider_U = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_U.setRange(500, 20000)
        self.slider_U.setValue(int(self.ctrl.get_voltage()))

        voltage_layout.addWidget(self.lbl_U)
        voltage_layout.addWidget(self.slider_U)
        left_layout.addWidget(voltage_box)

        # 3. УПРАВЛЕНИЕ СМЕСЯМИ (видно только в режиме смеси)
        self.mixture_box = QtWidgets.QGroupBox("Смесь газов")
        mixture_layout = QtWidgets.QVBoxLayout(self.mixture_box)

        self.combo_mixture = QtWidgets.QComboBox()
        mixtures = self.ctrl.get_mixture_list()
        self.combo_mixture.addItems(mixtures)
        self.combo_mixture.currentTextChanged.connect(self.on_mixture_changed)

        self.mixture_info = QtWidgets.QLabel("Выберите смесь...")
        self.mixture_info.setStyleSheet("""
            background-color: #252525; 
            padding: 6px; 
            border-radius: 3px; 
            color: white;
            border: 1px solid #2196F3;
            font-size: 10px;
        """)
        self.mixture_info.setWordWrap(True)

        mixture_layout.addWidget(QtWidgets.QLabel("Выберите смесь:"))
        mixture_layout.addWidget(self.combo_mixture)
        mixture_layout.addWidget(self.mixture_info)

        self.mixture_box.setVisible(False)  # Сначала скрываем
        left_layout.addWidget(self.mixture_box)

        # 4. УПРАВЛЕНИЕ ОДИНОЧНЫМ ЭЛЕМЕНТОМ (видно только в режиме single)
        self.single_box = QtWidgets.QGroupBox("Одиночный элемент")
        single_layout = QtWidgets.QVBoxLayout(self.single_box)

        self.unknown_info = QtWidgets.QLabel("Загрузка...")
        self.unknown_info.setStyleSheet("""
            background-color: #252525; 
            padding: 6px; 
            border-radius: 3px; 
            color: white;
            border: 1px solid #4CAF50;
            font-size: 10px;
        """)
        self.unknown_info.setWordWrap(True)

        self.input_guess = QtWidgets.QLineEdit()
        self.input_guess.setPlaceholderText("Символ элемента")

        button_row = QtWidgets.QHBoxLayout()
        self.btn_check = QtWidgets.QPushButton("Проверить")
        self.btn_new = QtWidgets.QPushButton("Новый")

        self.btn_check.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_new.setStyleSheet("background-color: #2196F3; color: white;")

        self.btn_check.clicked.connect(self.on_check_guess)
        self.btn_new.clicked.connect(self.on_new_unknown)

        button_row.addWidget(self.btn_check)
        button_row.addWidget(self.btn_new)

        single_layout.addWidget(self.unknown_info)
        single_layout.addWidget(QtWidgets.QLabel("Ваша догадка:"))
        single_layout.addWidget(self.input_guess)
        single_layout.addLayout(button_row)

        left_layout.addWidget(self.single_box)

        # 5. РЕФЕРЕНСНЫЕ ЭЛЕМЕНТЫ
        ref_box = QtWidgets.QGroupBox("Референсы")
        ref_layout = QtWidgets.QGridLayout(ref_box)
        ref_layout.setSpacing(1)

        self.ref_checkboxes = []
        for i, element in enumerate(self.reference_elements):
            checkbox = QtWidgets.QCheckBox(element)
            checkbox.setChecked(True)
            checkbox.setStyleSheet("font-size: 9px;")
            checkbox.stateChanged.connect(self.update_plots)
            self.ref_checkboxes.append(checkbox)
            row = i // 4
            col = i % 4
            ref_layout.addWidget(checkbox, row, col)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_all_ref = QtWidgets.QPushButton("Все")
        self.btn_none_ref = QtWidgets.QPushButton("Нет")
        self.btn_common_ref = QtWidgets.QPushButton("Частые")

        self.btn_all_ref.setStyleSheet("font-size: 9px; padding: 2px 5px;")
        self.btn_none_ref.setStyleSheet("font-size: 9px; padding: 2px 5px;")
        self.btn_common_ref.setStyleSheet("font-size: 9px; padding: 2px 5px;")

        self.btn_all_ref.clicked.connect(lambda: self.set_all_references(True))
        self.btn_none_ref.clicked.connect(lambda: self.set_all_references(False))
        self.btn_common_ref.clicked.connect(self.set_common_references)

        btn_row.addWidget(self.btn_all_ref)
        btn_row.addWidget(self.btn_none_ref)
        btn_row.addWidget(self.btn_common_ref)

        ref_layout.addLayout(btn_row, 3, 0, 1, 4)
        left_layout.addWidget(ref_box)

        # 6. НАСТРОЙКИ ГРАФИКОВ
        graph_box = QtWidgets.QGroupBox("Настройки")
        graph_layout = QtWidgets.QVBoxLayout(graph_box)

        self.check_show_comparison = QtWidgets.QCheckBox("Спектры сравнения")
        self.check_show_comparison.setChecked(False)
        self.check_show_comparison.stateChanged.connect(self.toggle_comparison)

        self.check_show_peaks = QtWidgets.QCheckBox("Метки пиков")
        self.check_show_peaks.setChecked(True)
        self.check_show_peaks.stateChanged.connect(self.update_plots)

        self.btn_highlight_peak = QtWidgets.QPushButton("Выделить пик")
        self.btn_highlight_peak.clicked.connect(self.highlight_nearest_peak)

        graph_layout.addWidget(self.check_show_comparison)
        graph_layout.addWidget(self.check_show_peaks)
        graph_layout.addWidget(self.btn_highlight_peak)
        left_layout.addWidget(graph_box)

        # 7. ЭЛЕМЕНТЫ ДЛЯ СРАВНЕНИЯ
        comp_box = QtWidgets.QGroupBox("Для сравнения")
        comp_layout = QtWidgets.QVBoxLayout(comp_box)

        comp_info = QtWidgets.QLabel(
            "Na-22.99  K-39.10\n"
            "Mg-24.31  Al-26.98\n"
            "Cu-63.55  C-12.01\n"
            "N-14.01   O-16.00"
        )
        comp_info.setStyleSheet("""
            font-family: monospace;
            font-size: 9px;
            padding: 4px;
        """)
        comp_layout.addWidget(comp_info)
        left_layout.addWidget(comp_box)

        # Растягивающий элемент
        left_layout.addStretch()

        # Устанавливаем контейнер в ScrollArea
        scroll_area.setWidget(left_container)
        main_layout.addWidget(scroll_area, 1)

        # === ПРАВАЯ ПАНЕЛЬ С ГРАФИКАМИ ===
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self.tabs = QtWidgets.QTabWidget()

        # Создаем панели для графиков
        self.panel_spec = MplPanel()
        self.panel_traj = MplPanel()

        self.tabs.addTab(self.panel_spec, "📊 Масс-спектр")
        self.tabs.addTab(self.panel_traj, "🛤️ Траектории")

        right_layout.addWidget(self.tabs)
        main_layout.addWidget(right_panel, 3)

        # Подключаем сигналы
        self.slider_U.valueChanged.connect(self.on_voltage_changed)

        # Подключаем события мыши
        self.panel_spec.canvas.mpl_connect('motion_notify_event', self.on_spec_hover)
        self.panel_spec.canvas.mpl_connect('button_press_event', self.on_spec_click)
        self.panel_traj.canvas.mpl_connect('motion_notify_event', self.on_traj_hover)

        # Устанавливаем начальный размер
        self.resize(1200, 700)

    def set_mode(self, mode):
        """Переключение режима работы"""
        self.ctrl.set_mode(mode)

        if mode == "single":
            self.btn_single.setChecked(True)
            self.btn_mixture.setChecked(False)
            self.single_box.setVisible(True)
            self.mixture_box.setVisible(False)
            self.ctrl.generate_new_unknown()
        elif mode == "mixture":
            self.btn_single.setChecked(False)
            self.btn_mixture.setChecked(True)
            self.single_box.setVisible(False)
            self.mixture_box.setVisible(True)
            # Устанавливаем первую смесь по умолчанию
            if self.combo_mixture.count() > 0:
                self.on_mixture_changed(self.combo_mixture.currentText())

        self.update_plots()

    def on_mixture_changed(self, mixture_name):
        """Обработчик изменения выбранной смеси"""
        self.ctrl.set_mixture(mixture_name)
        self.update_mixture_info()
        self.update_plots()

    def update_mixture_info(self):
        """Обновление информации о смеси"""
        info = self.ctrl.get_current_mixture_info()
        if info:
            components_text = ""
            for component in info["components"]:
                components_text += f"• {component['formula']}: m/z {component['mass']:.2f} ({component['intensity'] * 100:.1f}%)\n"

            self.mixture_info.setText(
                f"Смесь: {info['name']}\n"
                f"Компоненты:\n{components_text}"
            )

    def set_all_references(self, state):
        for checkbox in self.ref_checkboxes:
            checkbox.setChecked(state)
        self.update_plots()

    def set_common_references(self):
        common = ["H", "He", "C", "N", "O", "Na", "Mg", "Al"]
        for checkbox in self.ref_checkboxes:
            element = checkbox.text()
            checkbox.setChecked(element in common)
        self.update_plots()

    def toggle_comparison(self):
        self.comparison_visible = self.check_show_comparison.isChecked()
        self.update_plots()

    def on_voltage_changed(self, value):
        self.ctrl.set_voltage(value)
        self.lbl_U.setText(f"U = {value:.0f} V")
        self.update_plots()

    def on_check_guess(self):
        guess = self.input_guess.text().strip()
        if not guess:
            QtWidgets.QMessageBox.warning(self, "Внимание", "Введите символ элемента")
            return

        guess = guess[0].upper() + guess[1:].lower() if len(guess) > 1 else guess.upper()

        if self.ctrl.check_guess(guess):
            QtWidgets.QMessageBox.information(
                self, "🎉 Правильно!",
                f"Вы верно определили элемент {guess}!"
            )
            self.on_new_unknown()
        else:
            QtWidgets.QMessageBox.warning(self, "❌ Неверно", "Попробуйте еще раз.")

    def on_new_unknown(self):
        self.ctrl.generate_new_unknown()
        self.input_guess.clear()
        self.update_plots()

    def update_plots(self):
        self.update_spectrum()
        self.update_trajectories()

        # Обновляем информацию в зависимости от режима
        if self.ctrl.get_mode() == "single":
            self.update_unknown_info()
        elif self.ctrl.get_mode() == "mixture":
            self.update_mixture_info()

    def update_spectrum(self):
        try:
            for ann in self.peak_annotations:
                ann.remove()
            self.peak_annotations = []

            ax = self.panel_spec.ax
            ax.clear()

            ax.set_facecolor("#1A1A1A")
            ax.tick_params(colors="#E0E0E0", labelsize=8)
            ax.xaxis.label.set_color('#E0E0E0')
            ax.yaxis.label.set_color('#E0E0E0')
            ax.title.set_color('#E0E0E0')

            t, s = self.ctrl.spectrum()

            # В зависимости от режима, подписываем график по-разному
            if self.ctrl.get_mode() == "mixture":
                info = self.ctrl.get_current_mixture_info()
                if info:
                    ax.plot(t, s, color="#2196F3", linewidth=1.5, alpha=0.9,
                            label=f"Смесь: {info['name']}")
            else:
                ax.plot(t, s, color="#00BCD4", linewidth=1.5, alpha=0.9, label="Неизвестный")

            ax.fill_between(t, s, color="#00BCD4", alpha=0.1)

            self.add_reference_elements(ax, t, s)

            if self.comparison_visible:
                comparison_spectra = self.ctrl.get_comparison_spectra()
                for element_data in comparison_spectra[:4]:
                    t_comp, s_comp = element_data["data"]
                    ax.plot(t_comp, s_comp,
                            color=element_data["color"],
                            linestyle=element_data["style"],
                            linewidth=1.0,
                            alpha=element_data["alpha"],
                            label=element_data["name"])

            ax.set_xlabel("Время пролета, t (с)", fontsize=9)
            ax.set_ylabel("Интенсивность, I", fontsize=9)

            # Заголовок в зависимости от режима
            if self.ctrl.get_mode() == "mixture":
                ax.set_title("Масс-спектр смеси газов", fontsize=10, pad=6)
            else:
                ax.set_title("TOF Масс-спектр", fontsize=10, pad=6)

            ax.grid(True, alpha=0.1, color='#555', linestyle=':')

            legend = ax.legend(facecolor="#2A2A2A", edgecolor="#555",
                               labelcolor="#E0E0E0", fontsize=7, loc='upper right')
            legend.get_frame().set_alpha(0.9)

            if self.check_show_peaks.isChecked():
                self.mark_peaks(ax, t, s)

            self.panel_spec.canvas.draw()

        except Exception as e:
            print(f"Ошибка при обновлении спектра: {e}")

    def add_reference_elements(self, ax, t_data, s_data):
        try:
            from model.element_db import ELEMENTS

            selected_elements = []
            for checkbox in self.ref_checkboxes:
                if checkbox.isChecked():
                    element_name = checkbox.text()
                    if element_name in ELEMENTS:
                        selected_elements.append({
                            "name": element_name,
                            "mass": ELEMENTS[element_name]
                        })

            if not selected_elements:
                return

            colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0',
                      '#118AB2', '#EF476F', '#7209B7', '#F15BB5',
                      '#00BBF9', '#00F5D4', '#FB5607']

            ylim = ax.get_ylim()
            y_max = ylim[1]

            elements_by_time = []
            for i, element in enumerate(selected_elements):
                time = self.ctrl.tof.flight_time(element["mass"])
                elements_by_time.append({
                    "name": element["name"],
                    "mass": element["mass"],
                    "time": time,
                    "color": colors[i % len(colors)]
                })

            elements_by_time.sort(key=lambda x: x["time"])

            y_positions = []
            for element in elements_by_time:
                time = element["time"]
                color = element["color"]

                if time < np.min(t_data) or time > np.max(t_data):
                    continue

                ax.axvline(x=time, color=color, alpha=0.5,
                           linestyle=":", linewidth=1.0, zorder=1)

                text_y = y_max * 0.85
                for y_pos in y_positions:
                    pos_time, pos_y = y_pos
                    time_range = np.max(t_data) - np.min(t_data)
                    if abs(time - pos_time) < time_range * 0.05:
                        text_y = pos_y - (y_max * 0.05)
                        if text_y < y_max * 0.15:
                            text_y = y_max * 0.85

                ann = ax.text(time, text_y, element["name"],
                              color=color, fontsize=8, fontweight='bold',
                              ha='center', va='center',
                              bbox=dict(boxstyle="round,pad=0.1",
                                        facecolor="#2A2A2A",
                                        edgecolor=color,
                                        alpha=0.9),
                              zorder=10)
                self.peak_annotations.append(ann)

                y_positions.append((time, text_y))

                ax.plot([time, time], [y_max * 0.02, y_max * 0.03],
                        color=color, linewidth=1.0, alpha=0.6, zorder=2)

            ax.set_ylim(ylim)

        except Exception as e:
            print(f"Ошибка при добавлении референсов: {e}")

    def mark_peaks(self, ax, t, s):
        peaks = []
        for i in range(2, len(s) - 2):
            if (s[i] > s[i - 1] and s[i] > s[i - 2] and
                    s[i] > s[i + 1] and s[i] > s[i + 2] and
                    s[i] > 0.15 * np.max(s)):
                peaks.append((t[i], s[i]))

        for peak_time, peak_intensity in peaks:
            ax.axvline(x=peak_time, color='#FFEB3B', alpha=0.2, linestyle=':', linewidth=1.0)

            # Определяем, какой элемент соответствует пику
            element = self.ctrl.get_peak_at_time(peak_time)
            if element:
                label = f"{element.get('formula', element.get('name', ''))}\n{peak_time:.1e}s"
            else:
                label = f"{peak_time:.1e}s"

            ann = ax.annotate(
                label,
                xy=(peak_time, peak_intensity),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=7,
                color='#FFEB3B',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#333",
                          edgecolor="#FFEB3B", alpha=0.8)
            )
            self.peak_annotations.append(ann)

            ax.plot(peak_time, peak_intensity, 'o', color='#FFEB3B',
                    markersize=4, alpha=0.8, markeredgecolor='#FFF', markeredgewidth=1.0)

    def update_trajectories(self):
        try:
            trajectories = self.ctrl.get_all_trajectories()

            ax = self.panel_traj.ax
            ax.clear()

            ax.set_facecolor("#1A1A1A")
            ax.tick_params(colors="#E0E0E0", labelsize=8)
            ax.xaxis.label.set_color('#E0E0E0')
            ax.yaxis.label.set_color('#E0E0E0')
            ax.title.set_color('#E0E0E0')

            # Разные цвета для калибровочных элементов
            calib_colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0']

            # Сначала рисуем калибровочные элементы (тонкие линии)
            calib_count = 0
            for traj in trajectories:
                if traj["name"] in ["H", "He", "Si", "Fe"]:  # Калибровочные
                    v = traj["velocity"]
                    t_total = traj["time"]
                    tt = np.linspace(0, t_total, 100)
                    x = v * tt
                    color = calib_colors[calib_count % len(calib_colors)]
                    ax.plot(x, tt, traj["style"], color=color,
                            linewidth=1.0, label=traj["name"],
                            alpha=0.5)
                    calib_count += 1

            # Затем рисуем неизвестный элемент или смесь (толстые линии)
            for traj in trajectories:
                if traj["name"] not in ["H", "He", "Si", "Fe"]:  # Не калибровочные
                    v = traj["velocity"]
                    t_total = traj["time"]
                    tt = np.linspace(0, t_total, 150)
                    x = v * tt

                    # Используем радиус для толщины линии (если есть)
                    linewidth = traj.get("radius", 2.0)

                    if self.ctrl.get_mode() == "mixture":
                        # Для смеси показываем формулы
                        label = f"{traj.get('formula', traj['name'])}"
                        alpha = traj.get("intensity", 0.7) * 0.8 + 0.2
                    else:
                        # Для одиночного элемента
                        label = "Неизвестный" if traj["name"] == "Unknown" else traj["name"]
                        alpha = 0.9

                    ax.plot(x, tt, traj["style"], color=traj["color"],
                            linewidth=linewidth, label=label,
                            alpha=alpha, zorder=10)

            ax.set_xlabel("Расстояние, x (м)", fontsize=9)
            ax.set_ylabel("Время, t (с)", fontsize=9)

            # Заголовок в зависимости от режима
            if self.ctrl.get_mode() == "mixture":
                ax.set_title("Траектории ионов в смеси", fontsize=10, pad=6)
            else:
                ax.set_title("Траектории ионов", fontsize=10, pad=6)

            legend = ax.legend(facecolor="#2A2A2A", edgecolor="#555",
                               labelcolor="#E0E0E0", fontsize=7, loc='lower right')
            legend.get_frame().set_alpha(0.9)

            ax.grid(True, alpha=0.1, color='#555', linestyle=':')

            self.panel_traj.canvas.draw()

        except Exception as e:
            print(f"Ошибка при обновлении траекторий: {e}")

    def update_unknown_info(self):
        try:
            info = self.ctrl.get_unknown_info()
            if info and info.get("mode") == "single":
                mass_text = f"{info['mass']:.2f}"
                self.unknown_info.setText(
                    f"Неизвестный элемент\n"
                    f"Масса: {mass_text} а.е.м.\n"
                    f"Время: {info['time']:.1e} с"
                )
        except Exception as e:
            print(f"Ошибка при обновлении информации: {e}")

    def on_spec_hover(self, event):
        ax = self.panel_spec.ax

        if event.inaxes != ax:
            if self.spec_annotation:
                self.spec_annotation.set_visible(False)
                self.panel_spec.canvas.draw_idle()
            return

        t, intensity = event.xdata, event.ydata
        element = self.ctrl.get_peak_at_time(t)

        if self.spec_annotation:
            self.spec_annotation.remove()
            self.spec_annotation = None

        if element:
            mass = element["mass"]

            if self.ctrl.current_spectrum_data:
                t_data, s_data = self.ctrl.current_spectrum_data
                idx = np.argmin(np.abs(t_data - t))
                intensity = s_data[idx] if idx < len(s_data) else 0

            # Формируем текст в зависимости от режима
            if element["name"] == "Unknown":
                name_display = "Неизвестный элемент"
                color = "#FF5252"
                text = f"{name_display}\nm={mass:.2f}u\nt={element['time']:.1e}s"
            elif self.ctrl.get_mode() == "mixture":
                name_display = element.get("formula", element["name"])
                color = "#2196F3"
                intensity_percent = element.get("intensity", 0) * 100
                text = f"{name_display}\nm/z={mass:.2f}\nt={element['time']:.1e}s\n({intensity_percent:.1f}%)"
            else:
                name_display = element["name"]
                color = "#FFF"
                text = f"{name_display}\nm={mass:.2f}u\nt={element['time']:.1e}s"

            self.spec_annotation = ax.annotate(
                text,
                xy=(t, intensity),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", facecolor="#2A2A2A",
                          edgecolor=color, alpha=0.9, linewidth=1),
                fontsize=8,
                color="#E0E0E0",
                arrowprops=dict(arrowstyle="->", color=color, linewidth=1)
            )

            ax.plot(t, intensity, 'o', color=color, markersize=6, alpha=0.8,
                    markeredgecolor='#FFF', markeredgewidth=1, zorder=10)

            self.panel_spec.canvas.draw_idle()

    def on_spec_click(self, event):
        if event.inaxes != self.panel_spec.ax or event.button != 1:
            return

        t = event.xdata
        element = self.ctrl.get_peak_at_time(t)

        if element and element["name"] != "Unknown":
            QtWidgets.QMessageBox.information(
                self,
                "Информация о пике",
                f"Формула: {element.get('formula', element['name'])}\n"
                f"Масса: {element['mass']:.2f} u\n"
                f"Время пролета: {element['time']:.1e} с"
            )

    def highlight_nearest_peak(self):
        pos = self.panel_spec.canvas.mapFromGlobal(QtGui.QCursor.pos())

        if self.panel_spec.ax.contains(pos):
            x, y = pos.x(), pos.y()
            inv = self.panel_spec.ax.transData.inverted()
            data_coords = inv.transform((x, y))

            if data_coords:
                t = data_coords[0]
                element = self.ctrl.get_peak_at_time(t)

                if element:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Информация о пике",
                        f"Элемент: {element.get('formula', element['name'])}\n"
                        f"Масса: {element['mass']:.2f} u\n"
                        f"Время: {element['time']:.1e} с"
                    )

    def on_traj_hover(self, event):
        ax = self.panel_traj.ax

        if event.inaxes != ax:
            if self.traj_annotation:
                self.traj_annotation.set_visible(False)
                self.panel_traj.canvas.draw_idle()
            return

        x, t = event.xdata, event.ydata
        element = self.ctrl.get_element_at_point(x, t)

        if self.traj_annotation:
            self.traj_annotation.remove()
            self.traj_annotation = None

        if element:
            # Получаем информацию о траектории
            trajectories = self.ctrl.get_unknown_trajectory()
            traj_info = None
            for traj in trajectories:
                if traj["name"] == element:
                    traj_info = traj
                    break

            if traj_info:
                name = traj_info.get("formula", traj_info["name"])
                mass = traj_info["mass"]
                color = traj_info["color"]

                # Для смеси показываем интенсивность
                if self.ctrl.get_mode() == "mixture":
                    intensity = traj_info.get("intensity", 0) * 100
                    text = f"{name}\nm/z={mass:.2f}\nt={t:.1e}s\n({intensity:.1f}%)"
                else:
                    text = f"{name}\nm={mass:.2f}u\nt={t:.1e}s"

                self.traj_annotation = ax.annotate(
                    text,
                    xy=(x, t),
                    xytext=(8, 8),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", facecolor="#2A2A2A",
                              edgecolor=color, alpha=0.9, linewidth=1),
                    fontsize=8,
                    color="#E0E0E0",
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1)
                )

                ax.plot(x, t, 'o', color=color, markersize=6, alpha=0.8,
                        markeredgecolor='#FFF', markeredgewidth=1, zorder=10)

                self.panel_traj.canvas.draw_idle()