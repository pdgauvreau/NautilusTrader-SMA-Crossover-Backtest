"""Download Binance spot kline archives from the public data.binance.vision bucket.

Fetches one monthly zip per month of the requested year, verifies it against the
published ``.CHECKSUM`` (SHA256) file when one exists, and extracts the single
expected CSV into ``data/raw/``. Files already present are skipped unless
``--force`` is given.

Usage::

    python scripts/download_data.py                       # BTCUSDT 1d 2025
    python scripts/download_data.py --symbol ETHUSDT --interval 1h --year 2024
    python scripts/download_data.py --force

Only public archive files are downloaded; no API keys or accounts are used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw"
TIMEOUT_S = 30

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9]+$")


def archive_name(symbol: str, interval: str, year: int, month: int) -> str:
    """Return the archive base name, e.g. ``BTCUSDT-1d-2025-01``."""
    return f"{symbol}-{interval}-{year:04d}-{month:02d}"


def fetch(session: requests.Session, url: str) -> bytes | None:
    """GET ``url`` and return the body, or ``None`` on HTTP 404."""
    resp = session.get(url, timeout=TIMEOUT_S)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def parse_checksum(text: str, expected_file: str) -> str:
    """Extract the SHA256 hex digest from a Binance ``.CHECKSUM`` file body.

    The file format is ``<sha256>  <filename>``.
    """
    parts = text.strip().split()
    if not parts or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
        raise ValueError(f"Malformed checksum file for {expected_file}: {text[:100]!r}")
    if len(parts) > 1 and parts[1] != expected_file:
        raise ValueError(f"Checksum refers to {parts[1]!r}, expected {expected_file!r}")
    return parts[0].lower()


def extract_member(zip_bytes: bytes, member: str, out_dir: Path) -> Path:
    """Extract only ``member`` from the zip into ``out_dir``.

    The archive is treated as untrusted data: no other members are written and
    the output path is built from the expected name, never from archive metadata,
    so path traversal is impossible.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        if member not in zf.namelist():
            raise ValueError(f"Expected member {member!r} not in archive: {zf.namelist()}")
        data = zf.read(member)
    target = (out_dir / Path(member).name).resolve()
    if target.parent != out_dir.resolve():
        raise ValueError(f"Refusing to write outside {out_dir}: {target}")
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return target


def download_month(
    session: requests.Session,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    out_dir: Path,
    force: bool = False,
) -> str:
    """Download, verify and extract one monthly archive.

    Returns a short status string: ``"skipped"``, ``"downloaded"`` or ``"missing"``.
    """
    name = archive_name(symbol, interval, year, month)
    csv_name, zip_name = f"{name}.csv", f"{name}.zip"
    target = out_dir / csv_name
    if target.exists() and not force:
        return "skipped"

    url = f"{BASE_URL}/{symbol}/{interval}/{zip_name}"
    zip_bytes = fetch(session, url)
    if zip_bytes is None:
        return "missing"

    checksum_body = fetch(session, f"{url}.CHECKSUM")
    if checksum_body is not None:
        expected = parse_checksum(checksum_body.decode("utf-8", errors="replace"), zip_name)
        actual = hashlib.sha256(zip_bytes).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {zip_name}: {actual} != {expected}")
    else:
        print(f"  warning: no CHECKSUM published for {zip_name}; unverified", file=sys.stderr)

    extract_member(zip_bytes, csv_name, out_dir)
    return "downloaded"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--symbol", "--symbols", dest="symbols", nargs="+", default=["BTCUSDT"],
        help="One or more spot symbols (default: BTCUSDT)",
    )
    p.add_argument("--interval", default="1d", help="Kline interval (default: 1d)")
    p.add_argument("--year", type=int, default=None, help="Single calendar year (default: 2025)")
    p.add_argument("--start-year", type=int, default=None, help="First year of a range (inclusive)")
    p.add_argument("--end-year", type=int, default=None, help="Last year of a range (inclusive)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    p.add_argument("--force", action="store_true", help="Re-download files already present")
    args = p.parse_args(argv)
    for value in (*args.symbols, args.interval):
        if not _SAFE_TOKEN.match(value):
            p.error(f"symbol/interval must be alphanumeric, got {value!r}")
    if args.year is not None and (args.start_year or args.end_year):
        p.error("use either --year or --start-year/--end-year")
    if args.year is None and args.start_year is None and args.end_year is None:
        args.year = 2025
    if args.year is not None:
        args.start_year = args.end_year = args.year
    args.start_year = args.start_year or args.end_year
    args.end_year = args.end_year or args.start_year
    if args.start_year > args.end_year:
        p.error("--start-year must not be after --end-year")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code.

    Months before a symbol's first available archive are reported as
    "not listed yet" (e.g. SOLUSDT before its 2020 listing), not as errors.
    Missing months after data has started are errors.
    """
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "kline-archive-downloader/1.0"
        for symbol in args.symbols:
            seen_data = False
            not_listed = 0
            for year in range(args.start_year, args.end_year + 1):
                for month in range(1, 13):
                    name = archive_name(symbol, args.interval, year, month)
                    status = download_month(
                        session, symbol, args.interval, year, month, out_dir, args.force
                    )
                    if status == "missing" and not seen_data:
                        not_listed += 1
                        continue
                    seen_data = True
                    print(f"{name}: {status}")
                    if status == "missing":
                        missing.append(name)
            if not_listed:
                print(f"{symbol}: {not_listed} leading month(s) not listed yet, skipped")
            if not seen_data:
                missing.append(f"{symbol} (no data at all)")

    if missing:
        print(f"{len(missing)} archive(s) not available: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
