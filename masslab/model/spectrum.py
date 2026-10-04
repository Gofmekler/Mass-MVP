"""Построение TOF-спектра и поиск пиков."""
from dataclasses import dataclass

import numpy as np

PEAK_SIGMA = 15e-9   # с, ширина пика (временное разрешение детектора)
TIME_STEP = 3e-9     # с, шаг дискретизации регистратора


@dataclass(frozen=True)
class Peak:
    mass: float          # а.е.м. (z = +1)
    intensity: float     # относительная интенсивность
    label: str = ""      # подпись на графике; "" — без подписи


def time_window(tof, peaks, margin=1.15):
    """Верхняя граница шкалы времени: чуть дальше самого тяжёлого иона."""
    heaviest = max(p.mass for p in peaks)
    return tof.flight_time(heaviest) * margin


def build_spectrum(peaks, tof, t_max, rng, noise=0.02, sigma=PEAK_SIGMA, step=TIME_STEP):
    """Спектр как сумма гауссовых пиков + фон детектора; пики нормированы на 1.

    Фон: средний уровень noise с флуктуациями ±noise/2 (как у реального
    детектора), поэтому он виден и в логарифмической шкале.
    """
    n = int(t_max / step) + 1
    t = np.linspace(0.0, t_max, n)
    s = np.zeros_like(t)
    for p in peaks:
        t0 = tof.flight_time(p.mass)
        s += p.intensity * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))
    top = s.max()
    if top > 0:
        s /= top
    s += noise * (1.0 + 0.5 * rng.standard_normal(n))
    return t, np.clip(s, 0.0, None)


def find_peaks(t, s, sigma=PEAK_SIGMA, rel_threshold=0.03):
    """Положения пиков [(t, высота)], отсортированные по времени.

    Сигнал сглаживается окном ~σ, чтобы шум не давал ложных максимумов;
    из близких (< 2σ) максимумов остаётся самый высокий.
    """
    if len(t) < 3:
        return []
    dt = t[1] - t[0]
    k = max(1, int(round(sigma / dt)))
    smooth = np.convolve(s, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    background = float(np.median(smooth))
    threshold = background + rel_threshold * (smooth.max() - background)
    mid = smooth[1:-1]
    idx = np.where((mid > smooth[:-2]) & (mid >= smooth[2:]) & (mid > threshold))[0] + 1

    accepted = []
    for i in sorted(idx, key=lambda j: -smooth[j]):
        if all(abs(t[i] - t[j]) > 2 * sigma for j in accepted):
            accepted.append(i)
    return [(float(t[i]), float(smooth[i])) for i in sorted(accepted)]
