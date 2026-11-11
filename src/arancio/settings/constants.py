"""Relative filename and field paths used to persist and edit settings."""

# storage anchors this relative filename to its application-data root
ARANCIO_SETTINGS_FILENAME: str = "settings.yml"

# dotted paths of the fields in the settings file, used to edit one field
PERMISSIONS_FIELD: str = "permissions"
MODEL_FIELD: str = "model"
PROVIDER_FIELD: str = f"{MODEL_FIELD}.provider"
MODEL_NAME_FIELD: str = f"{MODEL_FIELD}.name"
REASONING_FIELD: str = f"{MODEL_FIELD}.reasoning"
REASONING_EFFORT_FIELD: str = f"{REASONING_FIELD}.effort"
REASONING_SUMMARY_FIELD: str = f"{REASONING_FIELD}.summary"
TURN_FIELD: str = "turn"
TURN_MAX_FIELD: str = f"{TURN_FIELD}.max"
RETRY_FIELD: str = f"{TURN_FIELD}.retry"
RETRY_MAX_FIELD: str = f"{RETRY_FIELD}.max"
RETRY_DELAY_FIELD: str = f"{RETRY_FIELD}.delay"
RETRY_MULTIPLIER_FIELD: str = f"{RETRY_FIELD}.multiplier"
