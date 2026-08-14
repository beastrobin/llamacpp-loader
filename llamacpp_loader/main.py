"""Application entry point. Bootstraps the Tkinter app."""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)


def main() -> None:
    import tkinter as tk
    from llamacpp_loader.gui.app import MainWindow

    root = tk.Tk()
    root.title("llamacpp-loader")

    # Save window state on close
    def on_closing():
        mw.save_window_state() if hasattr(mw, "save_window_state") else None
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)

    mw = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
