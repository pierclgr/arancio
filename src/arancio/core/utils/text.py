"""Text formatting utilities."""


def decapitalize(text: str) -> str:
    """Lowercase the first character of a string, leaving the rest untouched.

    Args:
        text: the string to decapitalize.

    Returns:
        ``text`` with its first character lowercased, or ``text`` unchanged
        if it is empty.
    """
    return text[:1].lower() + text[1:]
