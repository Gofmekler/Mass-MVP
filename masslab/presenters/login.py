MAX_LENGTH = 60


class LoginPresenter:
    def __init__(self, view, on_login):
        self._view = view
        self._on_login = on_login
        view.start_requested.connect(self._on_start)

    def start(self):
        self._view.reset()

    def _on_start(self):
        name = " ".join(self._view.student_name().split())
        group = " ".join(self._view.student_group().split())
        if not name or not group:
            self._view.show_error("Введите ФИО и номер группы")
            return
        if len(name) > MAX_LENGTH or len(group) > MAX_LENGTH:
            self._view.show_error(f"Слишком длинное значение (не более {MAX_LENGTH} символов)")
            return
        self._on_login(name, group)
