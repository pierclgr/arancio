"""Modal screen asking a question with preset answers plus a custom one."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class QuestionScreen(ModalScreen[str]):
    """Modal asking a question with preset answers plus a custom free-text one.

    Dismisses with the chosen answer's label, or with the typed text when the user
    submits the custom-answer input.
    """

    def __init__(self, question: str, answers: list[str]) -> None:
        """Store the question and its preset answers.

        Args:
            question: the question shown to the user.
            answers: the preset answers offered as buttons.
        """
        super().__init__()
        self._question = question
        self._answers = answers

    def compose(self) -> ComposeResult:
        """Build the modal's widgets.

        Yields:
            The dialog container with the question, the answer buttons and the
            custom-answer input.
        """
        with Vertical(id="question-dialog"):
            yield Label(self._question, id="question-text")
            with Horizontal(id="question-answers"):
                for index, answer in enumerate(self._answers):
                    yield Button(answer, id=f"answer-{index}")
            yield Input(placeholder="custom answer…", id="custom-answer")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the label of the pressed answer button.

        Args:
            event: the button-press event.
        """
        index = int(event.button.id.removeprefix("answer-"))
        self.dismiss(self._answers[index])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Dismiss with the typed custom answer when it is non-empty.

        Args:
            event: the input-submitted event.
        """
        event.stop()
        text = event.value.strip()
        if text:
            self.dismiss(text)
