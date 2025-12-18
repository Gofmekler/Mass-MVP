from model.tof_physics import TOFPhysics
from model.spectrum import generate_tof_spectrum

tof = TOFPhysics(voltage=3000, flight_length=1.2)

masses = [28.085, 55.845]  # Si, Fe
intens = [1.0, 0.6]

t, s = generate_tof_spectrum(masses, intens, tof)

print("TOF Si:", tof.flight_time(28.085))
print("TOF Fe:", tof.flight_time(55.845))
