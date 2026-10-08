"""Главный презентер: порядок этапов лабораторной работы."""
import time
from datetime import datetime

from masslab.model.session import STAGE_TITLES, STAGES, LabSession
from masslab.presenters.alloy import AlloyPresenter
from masslab.presenters.demo import DemoPresenter
from masslab.presenters.element import ElementPresenter
from masslab.presenters.formatting import attempts_text, duration
from masslab.presenters.login import LoginPresenter
from masslab.presenters.quiz import QuizPresenter
from masslab.presenters.report import ReportPresenter
from masslab.presenters.sandbox import SandboxPresenter
from masslab.presenters.theory import TheoryPresenter
from masslab.presenters.tour import TourPresenter

THEMES = ("dark", "light")
DEFAULT_THEME = "dark"
STAGE_CHIPS = ("Тест", "Задание 1", "Задание 2", "Задание 3", "Итог")
CLOSE_DURING_LAB = ("Лабораторная работа не завершена.\n"
                    "Результаты нигде не сохраняются и будут потеряны. Закрыть программу?")
CLOSE_ON_REPORT = ("Итоговый результат будет закрыт и нигде не сохранится.\n"
                   "Если нужен отчёт, сначала сохраните его в PDF. Закрыть программу?")
NEXT_STAGE = {"quiz": "task1", "task1": "task2", "task2": "task3", "task3": None}


class AppPresenter:
    def __init__(self, view, rng, np_rng, write_pdf, clock=time.monotonic, now=datetime.now):
        self._view = view
        self._clock = clock
        self._now = now
        self._session = None
        self._page = None
        self._login = LoginPresenter(view.login, self._on_login, self._on_sandbox,
                                     self._on_teacher_demo)
        self._stages = {
            "quiz": QuizPresenter(view.quiz, rng, lambda n, r: self._on_stage_done("quiz", n, r)),
            "task1": DemoPresenter(view.demo, rng, np_rng,
                                   lambda n, r: self._on_stage_done("task1", n, r)),
            "task2": ElementPresenter(view.element, rng, np_rng,
                                      lambda n, r: self._on_stage_done("task2", n, r)),
            "task3": AlloyPresenter(view.alloy, rng, np_rng,
                                    lambda n, r: self._on_stage_done("task3", n, r)),
        }
        self._report = ReportPresenter(view.report, self._on_new_session, self._on_exit,
                                       write_pdf)
        self._sandbox = SandboxPresenter(view.sandbox, np_rng, self.start)
        self._tour = TourPresenter(view)
        self._theory = TheoryPresenter(view.theory, np_rng)
        view.tick.connect(self._on_tick)
        view.theory_requested.connect(self._on_theory)
        view.help_requested.connect(self._on_help)
        view.theme_toggled.connect(self._on_theme_toggled)
        self._theme = DEFAULT_THEME          # только в памяти: не сохраняется между запусками
        view.set_theme(self._theme)

    @property
    def session(self):
        return self._session

    def start(self):
        self._session = None
        self._set_reveal(False)
        self._tour.reset()
        self._view.set_close_confirmation(None)
        self._view.set_timer("")
        self._update_stages()
        self._login.start()
        self._show_page("login")

    def _show_page(self, page):
        self._page = page
        self._view.show_page(page)

    def _on_login(self, name, group):
        self._session = LabSession(name, group, self._clock)
        self._view.set_close_confirmation(CLOSE_DURING_LAB)
        self._enter("quiz")

    def _set_reveal(self, enabled):
        for presenter in self._stages.values():
            presenter.reveal = enabled

    def _on_teacher_demo(self, name, group):
        self._session = LabSession(name, group, self._clock, teacher=True)
        self._set_reveal(True)
        self._view.set_close_confirmation(None)
        self._enter("quiz")

    def _on_sandbox(self):
        self._sandbox.start()
        self._view.set_timer("Режим песочницы")
        self._show_page("sandbox")

    def _enter(self, stage):
        self._session.start_stage(stage)
        self._stages[stage].start()
        self._show_page(stage)
        self._update_stages()
        self._on_tick()
        self._tour.start_if_new(stage)

    def _on_stage_done(self, stage, attempts, report):
        session = self._session
        session.finish_stage(stage, attempts, report)
        spent = duration(session.stage_duration(stage))
        next_stage = NEXT_STAGE[stage]
        if next_stage is None:
            self._show_report()
            return
        self._view.show_message(
            f"{STAGE_TITLES[stage]} — выполнено",
            f"Время: {spent}, {attempts_text(attempts)}.\n\n"
            f"Далее: {STAGE_TITLES[next_stage]}.")
        self._enter(next_stage)

    def _show_report(self):
        self._report.start(self._session, self._now())
        self._view.set_close_confirmation(CLOSE_ON_REPORT)
        self._show_page("report")
        self._update_stages()
        self._on_tick()

    def _on_new_session(self):
        if self._view.confirm("Новая сессия",
                              "Начать работу заново? Текущий результат будет удалён."):
            self.start()

    def _on_exit(self):
        self._view.close_app()

    def _on_theory(self):
        self._theory.open()

    @property
    def theme(self):
        return self._theme

    def _on_theme_toggled(self):
        self._theme = THEMES[(THEMES.index(self._theme) + 1) % len(THEMES)]
        self._view.set_theme(self._theme)

    def _on_help(self):
        if self._tour.has_tour(self._page):
            self._tour.start(self._page)
        else:
            self._view.show_message(
                "Подсказки",
                "Подсказки по элементам экрана доступны в заданиях.\n"
                "Теория и формулы — в «Методичке».")

    def _on_tick(self):
        s = self._session
        if s is None:
            return
        total = f"Всего {duration(s.total_duration())}"
        if s.teacher:
            total = "Демо · " + total
        if s.current is None:
            self._view.set_timer(total)
        else:
            self._view.set_timer(f"Этап {duration(s.stage_duration(s.current))} · {total}")

    def _update_stages(self):
        s = self._session
        states = []
        for i, title in enumerate(STAGE_CHIPS):
            if s is None:
                state = "pending"
            elif i < len(STAGES):
                record = s.records[STAGES[i]]
                if record.finished_at is not None:
                    state = "done"
                elif s.current == STAGES[i]:
                    state = "current"
                else:
                    state = "pending"
            else:
                state = "current" if s.completed else "pending"
            states.append((title, state))
        self._view.set_stages(states)
