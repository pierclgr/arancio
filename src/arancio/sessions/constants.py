"""Constants used to persist chat sessions."""

# persistent chat sessions live below the arancio working-directory root
ARANCIO_SESSIONS_DIR = "sessions"

# JSONL schema version stamped on every session, written by the recorder and
# validated by the manager when a session is read back
SESSION_FORMAT_VERSION = 1
