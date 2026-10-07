import hashlib

from masslab.config import SANDBOX_PASSWORD_SHA256

MAX_LENGTH = 60
TEACHER_NAME = "Преподаватель"
TEACHER_GROUP = "демонстрация"


class LoginPresenter:
    """Вход студента; скрытое поле с паролем открывает панель преподавателя."""

    def __init__(self, view, on_login, on_sandbox, on_teacher_demo):
        self._view = view
        self._on_login = on_login
        self._on_teacher_demo = on_teacher_demo
        view.start_requested.connect(self._on_start)
        view.secret_entered.connect(self._on_secret)
        view.sandbox_requested.connect(on_sandbox)
        view.teacher_demo_requested.connect(self._on_demo)

    def start(self):
        self._view.reset()
        self._view.show_teacher_panel(False)

    def _clean(self):
        return (" ".join(self._view.student_name().split()),
                " ".join(self._view.student_group().split()))

    def _on_start(self):
        name, group = self._clean()
        if not name or not group:
            self._view.show_error("Введите ФИО и номер группы")
            return
        if len(name) > MAX_LENGTH or len(group) > MAX_LENGTH:
            self._view.show_error(f"Слишком длинное значение (не более {MAX_LENGTH} символов)")
            return
        self._on_login(name, group)

    def _on_secret(self, text):
        self._view.clear_secret()
        if hashlib.sha256(text.encode("utf-8")).hexdigest() == SANDBOX_PASSWORD_SHA256:
            self._view.show_teacher_panel(True)

    def _on_demo(self):
        name, group = self._clean()
        self._on_teacher_demo((name or TEACHER_NAME)[:MAX_LENGTH],
                              (group or TEACHER_GROUP)[:MAX_LENGTH])
