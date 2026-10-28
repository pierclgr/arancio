"""Constants used by generic filesystem operations."""

# bounded reads keep large-file consumers from loading the whole file
READ_CHUNK_SIZE: int = 64 * 1024
