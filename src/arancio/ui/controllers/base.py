"""Abstract frontend controller contract, without a concrete UI dependency."""

from abc import ABC, abstractmethod

from arancio.core.events import LoginNotice
from arancio.core.messages.base import Message


class UIController(ABC):
    """Define questions, notices and display operations a frontend must supply.

    Questions block the calling worker, not a frontend event loop. Display calls finish
    their display work before returning. Cancellation must release a pending question
    with None and make later frontend requests harmless.
    """

    @abstractmethod
    def ask(self, question: str, answers: list[str]) -> str | None:
        """Wait for a preset answer, free text, or cancellation.

        Args:
            question: the text to display.
            answers: the preset answer labels.

        Returns:
            The selected label, free text, or None when cancelled.
        """

    @abstractmethod
    def cancel_questions(self) -> None:
        """Release a pending question with None without waiting for worker work."""

    @abstractmethod
    def show_login(self, login_notice: LoginNotice) -> None:
        """Display login details and return without waiting for sign-in.

        Args:
            login_notice: the provider, verification URL and user code.
        """

    @abstractmethod
    def reset_conversation(self, messages: list[Message]) -> None:
        """Replace the visible conversation with supplied replay messages.

        Args:
            messages: the messages to display, or an empty list for a new chat.
        """

    @abstractmethod
    def exit(self) -> None:
        """Request frontend closure; the caller must also invoke shutdown."""
