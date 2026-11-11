"""Entry point connecting the Textual frontend to the Application and running it."""

from arancio.application.base import Application
from arancio.ui.apps.tui import TUIApp
from arancio.ui.controllers.tui import TUIController


def main() -> None:
    """Build the controller, the Application and the app, then run the UI.

    The controller is built first (the Application needs it); the app it drives needs
    the Application, so ``ui_controller.app`` is assigned once the app exists. Every
    other backend layer is built by :meth:`Application.build`.
    """
    ui_controller = TUIController()
    application = Application.build(ui_controller)
    app = TUIApp(application)
    ui_controller.app = app
    # textual captures clicks/drags, so the terminal emulator's native text selection
    # needs its modifier key (e.g. Option in iTerm2, Fn in Terminal.app)
    app.run(mouse=True)


if __name__ == "__main__":
    main()
