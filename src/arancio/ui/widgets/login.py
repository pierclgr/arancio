"""Visible device-code login panel for the Textual app."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Link, Static


class LoginPanel(Vertical):
    """Display a provider's clickable verification URL and its device code."""

    def __init__(self, provider: str, verification_url: str, user_code: str) -> None:
        """Store the login details shown by the panel.

        Args:
            provider: display name of the provider asking for the sign-in.
            verification_url: browser address where the user enters the code.
            user_code: short code issued for the active sign-in attempt.
        """
        super().__init__(classes="login")
        self.__provider: str = provider
        self.__verification_url: str = verification_url
        self.__user_code: str = user_code

    def compose(self) -> ComposeResult:
        """Build the sign-in message, URL link and device code.

        Yields:
            Widgets that explain and complete the provider sign-in flow.
        """
        yield Static(f"Sign in to {self.__provider} to continue.")
        yield Static("Open this link:")
        yield Link(self.__verification_url, url=self.__verification_url)
        yield Static(f"Code: {self.__user_code}", classes="login-code")
        yield Static("After sign-in, this request continues automatically.")
