"""Построение TOF-спектра и поиск пиков."""
from dataclasses import dataclass

import numpy as np

TIME_STEP = 3e-9     # с, шаг дискретизации регистратора


@dataclass(frozen=True)
class Peak:
    mass: float              # а.е.м.
    intensity: float         # относительная интенсивность
    label: str = ""          # подпись на графике; "" — без подписи, "?" — неизвестный
    charge: int = 1
    background: bool = False  # пик остаточного газа

    @property
    def mz(self):
        return self.mass / self.charge


@dataclass(frozen=True)
class FoundPeak:
    time: float      # с
    height: float    # над фоном, отн. ед.
    fwhm: float      # полная ширина на полувысоте, с


# Остаточный газ в вакуумной камере: (подпись, масса, доля от главного пика)
RESIDUAL_GAS = (("H₂O⁺", 18.015, 0.04), ("N₂⁺", 28.013, 0.06), ("O₂⁺", 31.999, 0.02))


def residual_gas_peaks(scale=1.0):
    return [Peak(m, scale * share, label, background=True) for label, m, share in RESIDUAL_GAS]


def time_window(tof, peaks, margin=1.15):
    """Верхняя граница шкалы времени: чуть дальше самого медленного иона."""
    slowest = max(tof.flight_time(p.mass, p.charge) for p in peaks)
    return slowest * margin


def build_spectrum(peaks, tof, t_max, rng, noise=0.02, step=TIME_STEP):
    """Спектр как сумма гауссовых пиков + фон детектора; пики нормированы на 1.

    Ширина каждого пика определяется прибором (tof.peak_sigma). Фон детектора
    всегда положителен (флуктуации ±40 % около уровня noise, не ниже 0,2·noise),
    поэтому в логарифмической шкале он выглядит ровной полосой.
    """
    n = int(t_max / step) + 1
    t = np.linspace(0.0, t_max, n)
    s = np.zeros_like(t)
    for p in peaks:
        t0 = tof.flight_time(p.mass, p.charge)
        sigma = tof.sigma_at_time(t0, p.charge)
        s += p.intensity * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    top = s.max()
    if top > 0:
        s /= top
    s += noise * np.maximum(1.0 + 0.4 * rng.standard_normal(n), 0.2)
    return t, s


def find_peaks(t, s, width_at, rel_threshold=0.03):
    """Найденные пики [FoundPeak], отсортированные по времени.

    width_at(t) — ожидаемое σ пика в момент t. Сигнал сглаживается окном
    шириной ~σ (своим в каждой точке), а соседние максимумы без заметного
    провала между ними объединяются: шум на вершине широкого пика не даёт
    ложных пиков, а неразрешённые пики сливаются в один — как на приборе.
    """
    if len(t) < 3:
        return []
    smooth = _adaptive_smooth(t, s, width_at)
    background = float(np.median(smooth))
    threshold = background + rel_threshold * (smooth.max() - background)
    mid = smooth[1:-1]
    idx = np.where((mid > smooth[:-2]) & (mid >= smooth[2:]) & (mid > threshold))[0] + 1

    accepted = []
    for i in sorted(idx, key=lambda j: -smooth[j]):
        if all(_resolved(t, smooth, i, j, background, width_at) for j in accepted):
            accepted.append(i)
    return [FoundPeak(float(t[i]), float(smooth[i] - background),
                      _fwhm(t, smooth, i, background))
            for i in sorted(accepted)]


def _adaptive_smooth(t, s, width_at):
    """Скользящее среднее с полушириной окна ≈ σ/2 в каждой точке (через кумулятивную сумму)."""
    n = len(s)
    dt = t[1] - t[0]
    half = np.maximum(1, np.rint(width_at(t) / (2 * dt))).astype(int)
    cumsum = np.concatenate(([0.0], np.cumsum(s)))
    index = np.arange(n)
    lo = np.maximum(index - half, 0)
    hi = np.minimum(index + half + 1, n)
    return (cumsum[hi] - cumsum[lo]) / (hi - lo)


VALLEY_DEPTH = 0.25   # провал между пиками — не менее 25 % высоты меньшего пика


def _resolved(t, s, i, j, background, width_at):
    """Пик i (не выше пика j) считается отдельным, если между ними есть заметный провал."""
    if abs(t[i] - t[j]) < width_at(t[i]):
        return False
    a, b = sorted((i, j))
    valley = s[a:b + 1].min()
    return s[i] - valley >= VALLEY_DEPTH * (s[i] - background)


def _fwhm(t, s, i, background):
    half = background + (s[i] - background) / 2
    left = i
    while left > 0 and s[left] > half:
        left -= 1
    right = i
    while right < len(s) - 1 and s[right] > half:
        right += 1
    t_left = _cross(t, s, left, left + 1, half)
    t_right = _cross(t, s, right - 1, right, half)
    return max(t_right - t_left, t[1] - t[0])


def _cross(t, s, a, b, level):
    if s[b] == s[a]:
        return float(t[a])
    return float(t[a] + (level - s[a]) * (t[b] - t[a]) / (s[b] - s[a]))
