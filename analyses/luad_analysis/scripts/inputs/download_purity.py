"""Acquire the frozen Aran et al. purity workbook with an atomic source receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SOURCE_DIR, load_config, sha256_file  # noqa: E402

ARCHIVE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4671203/supplementaryFiles"
MEMBER = "ncomms9971-s2.xlsx"
EXPECTED_SHA256 = load_config()["sources"]["purity"]["sha256"]
EXPECTED_BYTES = 553082


def _validate_cached(path: Path, receipt_path: Path) -> None:
    """Reject orphaned or changed cached source files."""
    if not path.exists() or not receipt_path.exists():
        raise ValueError("orphan purity workbook or receipt requires explicit repair")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("url") != ARCHIVE_URL or receipt.get("archive_member") != MEMBER:
        raise ValueError("purity source receipt location changed")
    if sha256_file(path) != EXPECTED_SHA256 or receipt.get("sha256") != EXPECTED_SHA256:
        raise ValueError("cached purity workbook hash mismatch")
    if path.stat().st_size != EXPECTED_BYTES or int(receipt.get("bytes", -1)) != EXPECTED_BYTES:
        raise ValueError("cached purity workbook byte count mismatch")


def acquire(path: Path) -> None:
    """Download Supplementary Data 1 or validate its existing immutable bytes."""
    receipt_path = path.with_suffix(path.suffix + ".source.json")
    if path.exists() or receipt_path.exists():
        _validate_cached(path, receipt_path)
        logging.info("Verified cached purity workbook %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".purity-download-", dir=path.parent) as temporary:
        archive_path = Path(temporary) / "supplementary.zip"
        request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "CORAL-LUAD-composition/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("xb") as output:
            resolved_url = response.url
            content_type = response.headers.get("Content-Type")
            last_modified = response.headers.get("Last-Modified")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            if MEMBER not in archive.namelist():
                raise ValueError(f"supplementary archive does not contain {MEMBER}")
            extracted = Path(temporary) / MEMBER
            with archive.open(MEMBER) as source, extracted.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
        if extracted.stat().st_size != EXPECTED_BYTES or sha256_file(extracted) != EXPECTED_SHA256:
            raise ValueError("downloaded purity workbook differs from the frozen resource")
        receipt = {
            "url": ARCHIVE_URL,
            "resolved_url": resolved_url,
            "archive_member": MEMBER,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "article": "Systematic pan-cancer analysis of tumour purity",
            "doi": "10.1038/ncomms9971",
            "pmcid": "PMC4671203",
            "publication_year": 2015,
            "resource": "Supplementary Data 1: TCGA tumor purity estimates",
            "correction": "Column IHC denotes H&E staining per doi:10.1038/ncomms10707",
            "content_type": content_type,
            "last_modified": last_modified,
            "archive_sha256": sha256_file(archive_path),
            "sha256": EXPECTED_SHA256,
            "bytes": EXPECTED_BYTES,
        }
        receipt_tmp = Path(temporary) / "receipt.json"
        receipt_tmp.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(extracted, path)
        os.replace(receipt_tmp, receipt_path)
    logging.info("Acquired %s (%d bytes)", path, EXPECTED_BYTES)


def main() -> None:
    """Acquire only the approved Aran et al. Supplementary Data 1 workbook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE_DIR / MEMBER)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    acquire(args.output)


if __name__ == "__main__":
    main()
