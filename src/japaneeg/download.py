"""Download, verify and extract the JapanEEG release.

Access to the data is controlled. After your request on Harvard Dataverse
(https://doi.org/10.7910/DVN/5RM3OT) is approved under the JapanEEG Data Use Agreement, you
receive a user name and password for the data server. Pass them through the environment (or a
``.env`` file in the working directory), never on the command line::

    JAPANEEG_USER=<user name>
    JAPANEEG_PASSWORD=<password>

The server holds the BIDS tree as xz-compressed tar archives. ``archives_manifest.tsv`` lists
every archive with its size and SHA-256. :func:`download` fetches the manifest and the archives
(all of them, or those matching ``include``), resumes interrupted transfers, checks every
archive against the manifest and, with ``extract=True``, unpacks them into one BIDS tree.

Command line::

    uv run japaneeg-download --out data/JapanEEG                     # everything (0.37 TB)
    uv run japaneeg-download --out data/JapanEEG --include "sidecars/*" "archives/sub-01/sub-01_ses-20230829*"
"""

from __future__ import annotations

import argparse
import base64
import csv
import fnmatch
import hashlib
import os
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://eegdata.araya.org/JapanEEG/"
PUBLIC_FILES: tuple[str, ...] = (
    "README.md",
    "CHANGES",
    "dataset_description.json",
    "participants.tsv",
    "archives_manifest.tsv",
    "archive_files.tsv",
)
CHUNK = 1 << 20


@dataclass(frozen=True)
class Archive:
    """One entry of ``archives_manifest.tsv``.

    Attributes:
        path: Path relative to the release root, e.g. ``archives/sub-01/sub-01_ses-20230829.tar.xz``.
        bytes: Size in bytes.
        sha256: SHA-256 of the archive.
    """

    path: str
    bytes: int
    sha256: str


def load_dotenv(path: Path = Path(".env")) -> None:
    """Set ``KEY=value`` lines of a ``.env`` file as environment variables (existing ones win).

    Args:
        path: The ``.env`` file; nothing happens if it does not exist.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def credentials(user: str | None = None, password: str | None = None) -> tuple[str, str]:
    """User name and password from the arguments or ``JAPANEEG_USER`` / ``JAPANEEG_PASSWORD``.

    Args:
        user: User name (None: from the environment).
        password: Password (None: from the environment).

    Returns:
        The user name and password.

    Raises:
        SystemExit: If either is missing.
    """
    load_dotenv()
    user = user or os.environ.get("JAPANEEG_USER")
    password = password or os.environ.get("JAPANEEG_PASSWORD")
    if not user or not password:
        raise SystemExit("Set JAPANEEG_USER and JAPANEEG_PASSWORD (environment or .env); see the README.")
    return user, password


def _request(url: str, auth: tuple[str, str], start: int = 0) -> urllib.request.Request:
    """GET request with HTTP basic authentication and an optional byte range."""
    req = urllib.request.Request(url)
    token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    if start:
        req.add_header("Range", f"bytes={start}-")
    return req


def fetch(url: str, dest: Path, auth: tuple[str, str], size: int | None = None) -> None:
    """Download ``url`` to ``dest``, resuming a partial ``dest.part`` file.

    Args:
        url: File URL.
        dest: Destination path.
        auth: User name and password.
        size: Expected size in bytes (None: unknown); a complete ``dest`` is left as it is.
    """
    if dest.is_file() and (size is None or dest.stat().st_size == size):
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.is_file() else 0
    try:
        resp = urllib.request.urlopen(_request(url, auth, start), timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and size is not None and start == size:  # already complete
            part.rename(dest)
            return
        if e.code == 401:
            raise SystemExit("401 Unauthorized: check JAPANEEG_USER / JAPANEEG_PASSWORD (valid for 30 days).") from e
        raise
    mode = "ab" if start and resp.status == 206 else "wb"
    with resp, open(part, mode) as f:
        while chunk := resp.read(CHUNK):
            f.write(chunk)
    part.rename(dest)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> list[Archive]:
    """Entries of ``archives_manifest.tsv``.

    Args:
        path: The manifest.

    Returns:
        The archives.
    """
    with open(path, encoding="utf-8", newline="") as f:
        return [Archive(r["archive"], int(r["bytes"]), r["sha256"]) for r in csv.DictReader(f, delimiter="\t")]


def select(archives: list[Archive], include: list[str] | None) -> list[Archive]:
    """Archives whose path matches any of the glob patterns (all if ``include`` is empty).

    Args:
        archives: Manifest entries.
        include: Glob patterns on the archive path, e.g. ``"archives/sub-01/*"``.

    Returns:
        The selected archives.
    """
    if not include:
        return archives
    return [a for a in archives if any(fnmatch.fnmatch(a.path, p) for p in include)]


def extract(archive: Path, root: Path) -> None:
    """Unpack one release archive into the BIDS root (paths inside are relative to it).

    Args:
        archive: The ``.tar.xz`` file.
        root: BIDS root directory.
    """
    with tarfile.open(archive, "r:xz") as tar:
        tar.extractall(root, filter="data")


def download(
    out: str | Path = "data/JapanEEG",
    include: list[str] | None = None,
    extract_to: str | Path | None = None,
    user: str | None = None,
    password: str | None = None,
    base_url: str = BASE_URL,
) -> Path:
    """Download (and optionally extract) the release.

    Args:
        out: Download folder (archives are kept here with their release paths).
        include: Glob patterns of archives to fetch (None: all). The public files and the
            manifests are always fetched.
        extract_to: BIDS root to unpack the archives into (None: do not unpack).
        user: Data-server user name (None: ``JAPANEEG_USER``).
        password: Data-server password (None: ``JAPANEEG_PASSWORD``).
        base_url: Data-server URL.

    Returns:
        The download folder.

    Raises:
        SystemExit: If an archive does not match the manifest.
    """
    auth = credentials(user, password)
    out = Path(out)
    for name in PUBLIC_FILES:
        fetch(base_url + name, out / name, auth)
    archives = select(read_manifest(out / "archives_manifest.tsv"), include)
    total = sum(a.bytes for a in archives)
    print(f"{len(archives)} archives, {total / 1e9:.1f} GB", flush=True)
    for k, a in enumerate(archives, 1):
        dest = out / a.path
        fetch(base_url + a.path, dest, auth, a.bytes)
        if sha256_file(dest) != a.sha256:
            dest.rename(dest.with_name(dest.name + ".bad"))
            raise SystemExit(f"{a.path}: SHA-256 differs from the manifest (renamed to .bad); run again")
        print(f"[{k}/{len(archives)}] {a.path} ok", flush=True)
        if extract_to is not None:
            extract(dest, Path(extract_to))
    return out


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point (``japaneeg-download``)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/JapanEEG", help="download folder")
    ap.add_argument("--include", nargs="*", default=None, help="glob patterns of archive paths (default: all)")
    ap.add_argument("--extract-to", default=None, help="BIDS root to unpack into (default: <out>/bids)")
    ap.add_argument("--no-extract", action="store_true", help="download and verify only")
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args(argv)
    target = None if args.no_extract else (args.extract_to or str(Path(args.out) / "bids"))
    download(args.out, args.include, target, base_url=args.base_url)
    if target:
        print(f"BIDS tree: {target}")


if __name__ == "__main__":
    main(sys.argv[1:])
