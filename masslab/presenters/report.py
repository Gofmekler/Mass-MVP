import re

from masslab.config import FEEDBACK_FORM_URL
from masslab.model.session import STAGE_TITLES, STAGES
from masslab.presenters.formatting import attempts_text, duration
from masslab.views.viewmodels import ReportData, ReportRow

WORK_TITLE = "Лабораторная работа «Времяпролётный масс-спектрометр»"
FEEDBACK_TEXT = ("Пожалуйста, оцените приложение: отсканируйте QR-код и расскажите "
                 "о своём опыте использования. Анкета анонимная.")


class ReportPresenter:
    def __init__(self, view, on_new_session, on_exit, write_pdf):
        self._view = view
        self._write_pdf = write_pdf
        self._report = None
        view.new_session_requested.connect(on_new_session)
        view.exit_requested.connect(on_exit)
        view.export_requested.connect(self._on_export)

    def start(self, session, finished_at):
        rows = tuple(
            ReportRow(STAGE_TITLES[s], duration(session.stage_duration(s)),
                      attempts_text(session.records[s].attempts))
            for s in STAGES)
        self._report = ReportData(
            work_title=WORK_TITLE,
            student=session.student,
            group=session.group,
            finished_at=finished_at.strftime("%d.%m.%Y %H:%M"),
            rows=rows,
            total=duration(session.total_duration()),
            verdict=("ДЕМОНСТРАЦИЯ" if session.teacher
                     else "ЗАЧТЕНО" if session.completed else "НЕ ЗАВЕРШЕНО"),
            stages=tuple(session.reports[s] for s in STAGES if s in session.reports),
        )
        self._view.show_report(self._report)
        self._view.show_feedback_qr(FEEDBACK_FORM_URL, FEEDBACK_TEXT)
        self._view.show_export_result("", True)

    def default_file_name(self):
        r = self._report
        safe = re.sub(r"[^\w\-]+", "_", f"{r.student}_{r.group}").strip("_")
        return f"Отчёт_TOF_{safe}.pdf"

    def _on_export(self):
        path = self._view.ask_save_path(self.default_file_name())
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            self._write_pdf(path, self._report)
        except OSError as error:
            self._view.show_export_result(f"Не удалось сохранить отчёт: {error}", False)
            return
        self._view.show_export_result(f"Отчёт сохранён: {path}", True)
