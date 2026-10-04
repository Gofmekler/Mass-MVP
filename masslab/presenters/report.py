from masslab.model.session import STAGE_TITLES, STAGES
from masslab.presenters.formatting import attempts_text, duration
from masslab.views.viewmodels import ReportData, ReportRow


class ReportPresenter:
    def __init__(self, view, on_new_session, on_exit):
        self._view = view
        view.new_session_requested.connect(on_new_session)
        view.exit_requested.connect(on_exit)

    def start(self, session, finished_at):
        rows = tuple(
            ReportRow(STAGE_TITLES[s], duration(session.stage_duration(s)),
                      attempts_text(session.records[s].attempts))
            for s in STAGES)
        self._view.show_report(ReportData(
            student=session.student,
            group=session.group,
            finished_at=finished_at.strftime("%d.%m.%Y %H:%M"),
            rows=rows,
            total=duration(session.total_duration()),
            verdict="ЗАЧТЕНО" if session.completed else "НЕ ЗАВЕРШЕНО",
        ))
