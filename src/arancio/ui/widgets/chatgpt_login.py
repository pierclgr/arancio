"""Visible ChatGPT device-code login notice for the Textual app."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Link, Static


class ChatGPTLoginNotice(Vertical):
    """Display a clickable ChatGPT verification URL and its device code."""

    def __init__(self, verification_url: str, user_code: str) -> None:
        """Store the login details shown by the notice.

        Args:
            verification_url: browser address where the user enters the code.
            user_code: short code issued for the active sign-in attempt.
        """
        super().__init__(classes="chatgpt-login")
        self._verification_url = verification_url
        self._user_code = user_code

    def compose(self) -> ComposeResult:
        """Build the sign-in message, URL link and device code.

        Yields:
            Widgets that explain and complete the ChatGPT sign-in flow.
        """
        yield Static("Sign in to ChatGPT to continue.")
        yield Static("Open this link:")
        yield Link(self._verification_url, url=self._verification_url)
        yield Static(f"Code: {self._user_code}", classes="chatgpt-login-code")
        yield Static("After sign-in, this request continues automatically.")
