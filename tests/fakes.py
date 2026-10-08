"""Поддельные пассивные виды для тестирования презентеров без Qt."""
import hashlib

TEST_PASSWORD = "тестовый-пароль"
TEST_PASSWORD_SHA256 = hashlib.sha256(TEST_PASSWORD.encode("utf-8")).hexdigest()
from masslab.events import Event


class FakeView:
    """Записывает все вызовы методов; события создаются по списку имён."""

    def __init__(self, events=(), **returns):
        self.calls = []
        self.returns = dict(returns)
        for name in events:
            setattr(self, name, Event())

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args):
            self.calls.append((name, args))
            value = self.returns.get(name)
            return value() if callable(value) else value
        return method

    def last(self, name):
        for call, args in reversed(self.calls):
            if call == name:
                return args
        raise AssertionError(f"{name} не вызывался")

    def called(self, name):
        return any(call == name for call, _ in self.calls)


def workspace():
    return FakeView(["voltage_changed", "length_changed", "log_scale_toggled",
                     "spectrum_hovered"])


def task_view(*events, **returns):
    view = FakeView(events, **returns)
    view.workspace = workspace()
    return view


LAUNCHER_EVENTS = ("launch_requested", "double_charge_toggled")
LOGIN_EVENTS = ("start_requested", "secret_entered", "teacher_demo_requested",
                "sandbox_requested")


class FakeMainView(FakeView):
    def __init__(self):
        super().__init__(["tick", "theory_requested", "help_requested", "tour_next",
                          "tour_skip", "theme_toggled"], confirm=True)
        self.login = FakeView(LOGIN_EVENTS,
                              student_name="Иванов И. И.", student_group="ФИЗ-101")
        self.quiz = FakeView(["answer_selected", "next_requested", "prev_requested",
                              "finish_requested", "result_action_requested"])
        self.demo = task_view(*LAUNCHER_EVENTS, "answer_selected", "check_requested",
                              selected_elements=lambda: ["H", "Ar"],
                              double_charge_enabled=False)
        self.element = task_view("check_requested")
        self.alloy = task_view("elements_check_requested", "check_requested")
        self.report = FakeView(["new_session_requested", "exit_requested", "export_requested"])
        self.sandbox = task_view(*LAUNCHER_EVENTS, "gas_toggled", "exit_requested",
                                 selected_elements=lambda: ["H"], double_charge_enabled=True)
        self.theory = FakeView(["section_selected", "next_requested", "prev_requested",
                                "resolution_voltage_changed"])


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
