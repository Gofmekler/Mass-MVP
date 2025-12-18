import numpy as np

def gaussian(t, t0, I, sigma):
    return I * np.exp(-(t - t0)**2 / (2 * sigma**2))


def build_spectrum(ions, tof, t_min, t_max, n, sigma, noise):
    t = np.linspace(t_min, t_max, n)
    s = np.zeros_like(t)

    for ion, intensity in ions:
        t0 = tof.flight_time(ion.mass_u, ion.charge)
        s += gaussian(t, t0, intensity, sigma)

    s += noise * np.max(s) * np.random.randn(len(t))
    s[s < 0] = 0
    return t, s
