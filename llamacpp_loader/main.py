"""Application entry point. Bootstraps the Tkinter app."""

import logging
import os
import sys


def _setup_logging() -> None:
    """Log to %APPDATA%\\llamacpp-loader\\app.log and to stdout when available.

    When the app is launched via pythonw.exe there is no console, so a file
    handler is the only way to keep startup/runtime diagnostics accessible.
    """
    log_dir = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~")),
        "llamacpp-loader",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    # pythonw.exe has no stdout; only attach a console handler when one exists.
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
    )


_setup_logging()


def _show_fatal(msg: str) -> None:
    """Best-effort error dialog. Works even before the main root exists.

    Falls back silently if Tk cannot be initialized (e.g. headless), since the
    traceback is already written to the file handler in _setup_logging().
    """
    try:
        import tkinter as tk
        from tkinter import messagebox

        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("llamacpp-loader failed to start", msg)
        r.destroy()
    except Exception:
        pass


def main() -> None:
    import tkinter as tk
    from tkinter import messagebox
    from llamacpp_loader.gui.app import MainWindow

    try:
        root = tk.Tk()
    except Exception as exc:
        logging.exception("Failed to initialize Tk root")
        _show_fatal(f"Failed to initialize the GUI window:\n{exc}")
        return

    root.title("llamacpp-loader")
    # Minimum window size: prevents buttons from being squashed below readable width
    root.minsize(960, 720)

    # Window-close cleanup (save state + stop server) is handled by
    # MainWindow._on_closing, registered in MainWindow.__init__.

    try:
        mw = MainWindow(root)
    except Exception as exc:
        logging.exception("Failed to build MainWindow")
        try:
            messagebox.showerror(
                "llamacpp-loader failed to start",
                f"{exc}\n\n(See %APPDATA%\\llamacpp-loader\\app.log for the full traceback)",
            )
        except Exception:
            pass
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Last-resort safety net: anything that escapes main() is still logged
        # and surfaced instead of being swallowed by pythonw.exe.
        logging.exception("Unhandled exception during startup")
        _show_fatal("Startup failed — see %APPDATA%\\llamacpp-loader\\app.log for details")
