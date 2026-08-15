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


def main() -> None:
    import tkinter as tk
    from llamacpp_loader.gui.app import MainWindow

    root = tk.Tk()
    root.title("llamacpp-loader")
    # Minimum window size: prevents buttons from being squashed below readable width
    root.minsize(960, 720)

    # Save window state on close
    def on_closing():
        mw.save_window_state() if hasattr(mw, "save_window_state") else None
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)

    mw = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
