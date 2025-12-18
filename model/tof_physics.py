import numpy as np

E_CHARGE = 1.602176634e-19
AMU = 1.66053906660e-27


class TOFPhysics:
    def __init__(self, voltage, flight_length):
        self.U = voltage
        self.L = flight_length

    def velocity(self, mass_u, charge=1):
        return (2 * charge * E_CHARGE * self.U / (mass_u * AMU)) ** 0.5

    def flight_time(self, mass_u, charge=1):
        return self.L / self.velocity(mass_u, charge)

    def mass_from_time(self, t, charge=1):
        return (2 * charge * E_CHARGE * self.U * t**2 / self.L**2) / AMU
