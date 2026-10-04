"""Форматирование чисел и времени для интерфейса (десятичная запятая)."""
from masslab.model.elements import ELEMENTS


def num(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


def parse_number(text):
    """'22,9' или '22.9' → 22.9; некорректный ввод → None."""
    try:
        value = float(text.strip().replace(",", ".").replace(" ", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def duration(seconds):
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def attempts_text(n):
    if n % 10 == 1 and n % 100 != 11:
        word = "попытка"
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        word = "попытки"
    else:
        word = "попыток"
    return f"{n} {word}"


def reference_rows(with_isotopes=False):
    """Строки справочника: символ, название, масса[, изотопы]."""
    rows = []
    for e in sorted(ELEMENTS.values(), key=lambda e: e.mass):
        row = [e.symbol, e.name, num(e.mass, 3)]
        if with_isotopes:
            row.append(", ".join(f"{i.mass_number} ({num(i.abundance * 100, 1)} %)"
                                 for i in e.isotopes) or "—")
        rows.append(row)
    return rows
