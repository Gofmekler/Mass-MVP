from model.tof_physics import TOFPhysics
from model.spectrum import build_spectrum
from model.unknown import UnknownSample


class SimulationController:
    def __init__(self):
        self.tof = TOFPhysics(3000, 1.2)
        self.sample = UnknownSample()

    def new_sample(self, mixture=False):
        self.sample = UnknownSample(mixture)

    def set_voltage(self, U):
        self.tof.U = U

    def spectrum(self):
        return build_spectrum(
            self.sample.ions(),
            self.tof,
            t_min=0,
            t_max=2e-5,
            n=4000,
            sigma=3e-8,
            noise=0.02
        )
