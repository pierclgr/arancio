"""Generic filesystem operations anchored to the application-data root."""

import errno
import os
import shutil
from collections.abc import Generator
from pathlib import Path

from arancio.core.constants import ARANCIO_DEFAULT_DIR
from arancio.storage.constants import READ_CHUNK_SIZE


class StorageManager:
    """Manage arancio's persistent on-disk storage.

    Relative paths are anchored to the application root. Absolute paths are
    used as supplied. Callers choose paths and own serialization and policy.

    Attributes:
        __root: the absolute application-data directory anchoring relative paths.
    """

    def __init__(self) -> None:
        """Store the application-data root without creating anything."""
        self.__root: Path = ARANCIO_DEFAULT_DIR

    def __repr__(self) -> str:
        """Return a developer-friendly representation of the manager.

        Returns:
            The manager's class name followed by its working directory root.
        """
        return f"{type(self).__name__}({self.__root})"

    @property
    def root(self) -> Path:
        """Return the absolute base directory used for relative paths.

        Returns:
            The application-data directory.
        """
        return self.__root

    def make_dir(self, path: str | Path) -> None:
        """Create a directory and its missing parents.

        Args:
            path: the directory path. Relative paths are resolved under the
                application root.
        """
        directory: Path = self.__resolve_path(path)
        directory.mkdir(parents=True, exist_ok=True)

    def find_files(self, directory: str | Path, pattern: str) -> list[Path]:
        """Return files below a directory matching a glob pattern.

        Args:
            directory: the directory tree to search.
            pattern: the filename glob pattern to match.

        Returns:
            Matching paths in deterministic path order.
        """
        return sorted(self.__resolve_path(directory).rglob(pattern))

    def read_text(self, path: str | Path) -> str:
        """Read a whole UTF-8 file as one string, including its line breaks.

        Args:
            path: the text file to read.

        Returns:
            The complete text, or an empty string for an empty file.
        """
        return self.__resolve_path(path).read_text(encoding="utf-8")

    def read_lines(self, path: str | Path) -> list[str]:
        """Read a UTF-8 text file as raw lines.

        Args:
            path: the text file to read.

        Returns:
            The file's lines without their terminating newlines.
        """
        return self.read_text(path).splitlines()

    def iter_file_bytes(self, path: str | Path) -> Generator[bytes, None, None]:
        """Read a file in bounded raw-byte chunks.

        Close the iterator when abandoning it before exhaustion. The file
        also closes when iteration completes or a read fails.

        Args:
            path: the file to read without decoding its contents.

        Yields:
            Byte chunks no larger than :data:`READ_CHUNK_SIZE`.
        """
        with self.__resolve_path(path).open("rb") as file:
            while True:
                chunk: bytes = file.read(READ_CHUNK_SIZE)
                if not chunk:
                    return
                yield chunk

    def file_size(self, path: str | Path) -> int:
        """Return the current byte length of a file.

        Args:
            path: the file to inspect.

        Returns:
            The file's current size in bytes.
        """
        return self.__resolve_path(path).stat().st_size

    def append_line(self, path: str | Path, line: str) -> None:
        """Append and sync one UTF-8 text line.

        Args:
            path: the text file to append.
            line: text without a trailing newline.
        """
        resolved_path: Path = self.__resolve_path(path)
        self.make_dir(resolved_path.parent)
        with resolved_path.open("ab") as file:
            file.write(line.encode("utf-8") + b"\n")
            file.flush()
            os.fsync(file.fileno())

    def create_file(self, path: str | Path, line: str | None = None) -> None:
        """Create and sync a new UTF-8 text file, optionally with its first line.

        Args:
            path: the new file path, which must not already exist.
            line: optional text without a trailing newline.
        """
        resolved_path: Path = self.__resolve_path(path)
        self.make_dir(resolved_path.parent)
        with resolved_path.open("xb") as file:
            if line is not None:
                file.write(line.encode("utf-8") + b"\n")
            file.flush()
            os.fsync(file.fileno())

    def truncate_file(self, path: str | Path, offset: int) -> None:
        """Discard every byte in a file after ``offset``.

        Args:
            path: the file to truncate.
            offset: the retained byte length from the beginning of the file.
        """
        with self.__resolve_path(path).open("r+b") as file:
            file.truncate(offset)
            file.flush()
            os.fsync(file.fileno())

    def remove_file(self, path: str | Path) -> None:
        """Remove a file.

        Args:
            path: the file to remove.
        """
        self.__resolve_path(path).unlink()

    def write_file(self, path: str | Path, content: str) -> None:
        """Create parent directories and replace a UTF-8 text file's content.

        Args:
            path: the text file to create or replace.
            content: the complete UTF-8 text content to write.
        """
        resolved_path: Path = self.__resolve_path(path)
        self.make_dir(resolved_path.parent)
        resolved_path.write_text(content, encoding="utf-8")

    def move_directory(self, source: str | Path, target: str | Path) -> None:
        """Move directory contents into a target and remove the emptied source.

        Reuse an existing target directory. Entry-name conflicts follow
        ``shutil.move`` behavior, and a failed migration is not rolled back.

        Args:
            source: the existing source directory, which must not be a symlink.
            target: the destination directory to create or reuse.
        """
        source_path: Path = self.__resolve_path(source)
        target_path: Path = self.__resolve_path(target)
        self.__validate_directory(source_path)
        self.make_dir(target_path)

        for entry in source_path.iterdir():
            shutil.move(str(entry), str(target_path / entry.name))
        source_path.rmdir()

    def create_symlink(self, source: str | Path, target: str | Path) -> None:
        """Create a directory symlink at source pointing to an existing target.

        Validate the target before checking the source. An existing source
        symlink resolving to the same target is left untouched. Every other
        occupied source path raises FileExistsError without replacement.

        Args:
            source: the symlink location, whose missing parents are created.
            target: the existing directory, which must not be a symlink.
        """
        source_path: Path = self.__resolve_path(source)
        target_path: Path = self.__resolve_path(target)
        self.__validate_directory(target_path)

        if source_path.is_symlink() and source_path.resolve() == target_path.resolve():
            return

        self.make_dir(source_path.parent)
        source_path.symlink_to(target_path, target_is_directory=True)

    def __resolve_path(self, path: str | Path) -> Path:
        """Anchor relative paths to the root without following symlinks.

        Args:
            path: the relative or absolute path supplied by a caller.

        Returns:
            The absolute path used for the operation.
        """
        resolved_path: Path = Path(path)
        if resolved_path.is_absolute():
            return resolved_path
        return self.__root / resolved_path

    @staticmethod
    def __validate_directory(path: Path) -> None:
        """Require an existing directory that is not itself a symlink.

        Args:
            path: the absolute directory path to validate.

        Raises:
            ValueError: when the path is a symlink.
            FileNotFoundError: when the path does not exist.
            NotADirectoryError: when the path is not a directory.
        """
        if path.is_symlink():
            raise ValueError(f"Expected a directory, got a symlink: {path}")
        if not path.exists():
            raise FileNotFoundError(errno.ENOENT, "Directory does not exist", path)
        if not path.is_dir():
            raise NotADirectoryError(errno.ENOTDIR, "Expected a directory", path)
