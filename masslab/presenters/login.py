import hashlib

from masslab.config import SANDBOX_PASSWORD_SHA256

MAX_LENGTH = 60


class LoginPresenter:
    def __init__(self, view, on_login, on_sandbox):
        self._view = view
        self._on_login = on_login
        self._on_sandbox = on_sandbox
        view.start_requested.connect(self._on_start)
        view.secret_entered.connect(self._on_secret)

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

    def _on_secret(self, text):
        self._view.clear_secret()
        if hashlib.sha256(text.encode("utf-8")).hexdigest() == SANDBOX_PASSWORD_SHA256:
            self._on_sandbox()
