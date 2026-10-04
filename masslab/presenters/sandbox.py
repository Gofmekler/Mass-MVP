from masslab.presenters.formatting import reference_rows
from masslab.presenters.launcher import IonLauncher
from masslab.presenters.workspace import WorkspacePresenter


class SandboxPresenter:
    """Песочница преподавателя: прибор со всеми настройками, без заданий."""

    def __init__(self, view, np_rng, on_exit):
        self._view = view
        self._ws = WorkspacePresenter(view.workspace, np_rng, noise=0.01,
                                      show_mass=True, adjustable_length=True)
        self._launcher = IonLauncher(view, self._ws)
        view.gas_toggled.connect(self._ws.set_residual_gas)
        view.exit_requested.connect(on_exit)

    def start(self):
        self._ws.reset()
        self._ws.set_residual_gas(True)
        self._view.set_gas(True)
        self._launcher.reset()
        self._view.show_reference(reference_rows(with_isotopes=True))
        self._view.show_feedback("", None)
