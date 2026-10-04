"""Поддельные пассивные виды для тестирования презентеров без Qt."""
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
    return FakeView(["voltage_changed", "log_scale_toggled", "spectrum_hovered"])


def task_view(*events, **returns):
    view = FakeView(events, **returns)
    view.workspace = workspace()
    return view


class FakeMainView(FakeView):
    def __init__(self):
        super().__init__(["tick"], confirm=True)
        self.login = FakeView(["start_requested"], student_name="Иванов И. И.",
                              student_group="ФИЗ-101")
        self.quiz = FakeView(["answer_selected", "next_requested", "prev_requested",
                              "finish_requested", "result_action_requested"])
        self.demo = task_view("launch_requested", "answer_selected", "check_requested")
        self.element = task_view("check_requested")
        self.alloy = task_view("check_requested")
        self.report = FakeView(["new_session_requested", "exit_requested"])


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
