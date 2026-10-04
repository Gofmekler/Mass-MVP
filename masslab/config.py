"""Настройки, которые может поменять автор работы."""

# Ссылка на анонимную анкету об опыте использования (показывается QR-кодом
# на итоговом экране). Замените на ссылку своей Google-формы.
FEEDBACK_FORM_URL = "https://forms.gle/ZAMENITE-NA-SVOYU-FORMU"

# SHA-256 пароля режима песочницы. В коде хранится только хэш, чтобы пароль
# нельзя было прочитать в исходниках. Новый хэш:
#   python -c "import hashlib; print(hashlib.sha256('пароль'.encode()).hexdigest())"
SANDBOX_PASSWORD_SHA256 = "46db8f98516613ca2397b9dd4b4fcc92efb5ce07dba8d18c12b1a9d22596a751"
