"""Constants used to persist chat sessions."""

# persistent chat sessions live below the arancio working-directory root
ARANCIO_SESSIONS_DIR = "sessions"

# JSONL schema version stamped on every session, written by the recorder and
# validated by the validator when a session is read back
SESSION_FORMAT_VERSION = 1

# SHA-256 of each session log is stored in a sibling file with this suffix
SESSION_CHECKSUM_SUFFIX = ".sha256"

# preserve the existing readable directory naming limits
SESSION_DIRECTORY_SLUG_MAX_LENGTH = 180
SESSION_DIRECTORY_HASH_LENGTH = 16

# strip one trailing lineage suffix before naming a fork
SESSION_LINEAGE_SUFFIX_PATTERN = r":(main|fork_\d{14})$"
