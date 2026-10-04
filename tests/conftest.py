import pytest

from tests.fakes import TEST_PASSWORD_SHA256


@pytest.fixture(autouse=True)
def test_sandbox_password(monkeypatch):
    """Тесты не знают настоящий пароль песочницы — подставляем хэш тестового."""
    monkeypatch.setattr("masslab.presenters.login.SANDBOX_PASSWORD_SHA256", TEST_PASSWORD_SHA256)
