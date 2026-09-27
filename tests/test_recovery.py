"""Boot-recovery tests: torn or corrupt state files are quarantined on startup."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cranefleet.console import Request
from cranefleet.errors import StoreError
from cranefleet.recovery import QUARANTINE_SUFFIX, quarantine_file, recover_store
from cranefleet.store import FileStore, decode_snapshot, encode_snapshot
from cranefleet.wiring import build_app

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def corrupt_names(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir() if QUARANTINE_SUFFIX in entry.name)


def post_config(console, body: dict) -> None:  # noqa: ANN001, ANN202
    request = Request(method="POST", path="/api/config", query={}, body=body, body_valid=True)
    console.dispatch(request)


class RecoverStoreTests(unittest.TestCase):
    def test_torn_snapshot_is_quarantined_and_key_state_survives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FileStore(directory)
            store.save("zone/zone-1", b'{"revision": 3}')
            snapshot_path = Path(directory) / "snapshot.json"
            snapshot_path.write_bytes(encode_snapshot({"zone/zone-1": b'{"revision": 9}'})[:31])

            report = recover_store(store, snapshot_path)

            self.assertEqual(report.snapshot_state, "quarantined")
            self.assertFalse(snapshot_path.exists())
            quarantined = corrupt_names(Path(directory))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue(quarantined[0].startswith("snapshot.json" + QUARANTINE_SUFFIX))
            self.assertIn("not valid JSON", report.quarantined[0].reason)
            self.assertEqual(store.load("zone/zone-1"), b'{"revision": 3}')
            self.assertIn("per-key store state", report.summary())

    def test_valid_snapshot_is_restored_and_left_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FileStore(directory)
            snapshot_path = Path(directory) / "snapshot.json"
            snapshot_path.write_bytes(encode_snapshot({"config/site": b'{"site_name": "x"}'}))

            report = recover_store(store, snapshot_path)

            self.assertEqual(report.snapshot_state, "restored")
            self.assertEqual(report.quarantined, ())
            self.assertTrue(snapshot_path.exists())
            self.assertEqual(store.load("config/site"), b'{"site_name": "x"}')

    def test_missing_snapshot_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = recover_store(FileStore(directory), Path(directory) / "snapshot.json")
            self.assertEqual(report.snapshot_state, "absent")
            self.assertEqual(report.quarantined, ())

    def test_corrupt_key_file_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FileStore(directory)
            store.save("config/site", b'{"site_name": "good"}')
            Path(directory, "zone_zone-1.bin").write_bytes(b'{"revision": 3,')
            Path(directory, "audit_entries.bin").write_bytes(b"\xff\xfe not utf-8")

            report = recover_store(store, Path(directory) / "snapshot.json")

            self.assertEqual(sorted(item.source.name for item in report.quarantined),
                             ["audit_entries.bin", "zone_zone-1.bin"])
            self.assertEqual(store.keys(), ["config/site"])
            self.assertEqual(len(corrupt_names(Path(directory))), 2)

    def test_stale_temporary_files_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "snapshot.json.tmp").write_bytes(b"half written")
            Path(directory, "zone_zone-1.bin.tmp").write_bytes(b"")

            report = recover_store(FileStore(directory), Path(directory) / "snapshot.json")

            self.assertEqual(len(report.stale_temporary), 2)
            self.assertFalse(any(Path(directory).glob("*.tmp")))

    def test_quarantine_never_overwrites_an_earlier_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for content in (b"bad one", b"bad two"):
                path = Path(directory, "snapshot.json")
                path.write_bytes(content)
                quarantine_file(path, "test")
            self.assertEqual(len(corrupt_names(Path(directory))), 2)


class SnapshotDecodingTests(unittest.TestCase):
    def test_rejects_torn_and_malformed_payloads(self) -> None:
        bad_payloads = [
            b"\xff\xff",  # not UTF-8
            b'{"version": 1, "da',  # torn JSON
            b"[1, 2, 3]",  # not an object
            b'{"version": 1}',  # missing data
            b'{"version": 2, "data": {}}',  # unsupported version
            b'{"version": 1, "data": {"a": 7}}',  # non-string entry
            b'{"version": 1, "data": {"a": "!!not-base64!!"}}',
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(StoreError):
                    decode_snapshot(payload)

    def test_round_trip_still_works(self) -> None:
        encoded = encode_snapshot({"a": b"\x00\x01", "b": b"text"})
        self.assertEqual(decode_snapshot(encoded), {"a": b"\x00\x01", "b": b"text"})


class BootRecoveryTests(unittest.TestCase):
    def test_service_boots_from_key_files_when_snapshot_is_torn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = build_app(directory, web_root=str(WEB_ROOT))
            post_config(first.console, {"site_name": "华东六号", "safe_separation": 20})
            first.shutdown()
            snapshot_path = Path(directory) / "snapshot.json"
            payload = snapshot_path.read_bytes()
            snapshot_path.write_bytes(payload[: len(payload) // 2])  # power cut mid-write

            second = build_app(directory, web_root=str(WEB_ROOT))

            self.assertEqual(second.console.config.site_name, "华东六号")
            self.assertFalse(snapshot_path.exists())
            self.assertEqual(len(corrupt_names(Path(directory))), 1)
            recovery = [e for e in second.console.audit.snapshot() if e.action == "recovery"]
            self.assertEqual(len(recovery), 1)
            self.assertIn("snapshot.json", recovery[0].detail)
            second.shutdown()

    def test_service_still_boots_when_nothing_is_usable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "snapshot.json").write_bytes(b'{"version": 1, "data":')
            Path(directory, "zone_zone-1.bin").write_bytes(b"garbage")

            app = build_app(directory, web_root=str(WEB_ROOT))

            self.assertFalse(app.console.zone.is_durable())
            self.assertEqual(len(corrupt_names(Path(directory))), 2)
            app.shutdown()
            # The fresh shutdown writes a clean snapshot, so the next boot restores it.
            self.assertTrue(json.loads((Path(directory) / "snapshot.json").read_text()))

    def test_clean_shutdown_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = build_app(directory, web_root=str(WEB_ROOT))
            app.shutdown()
            self.assertFalse(any(Path(directory).glob("*.tmp")))
            self.assertEqual(corrupt_names(Path(directory)), [])


if __name__ == "__main__":
    unittest.main()
