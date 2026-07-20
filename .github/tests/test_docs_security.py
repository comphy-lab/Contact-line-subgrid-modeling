#!/usr/bin/env python3
"""Security and failure-semantics tests for the documentation site."""

import hashlib
import importlib.util
import io
import re
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / ".github" / "scripts" / "generate_docs.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_docs_test_module", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load documentation generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_generator()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.payload


def make_tar_gz(files):
    """Return a gzip-compressed tar containing the given name/content pairs."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        for name, content in files.items():
            data = content.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def make_link_tar_gz(name, target, link_type):
    """Return a gzip-compressed tar containing one link member."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.type = link_type
        member.linkname = target
        bundle.addfile(member)
    return stream.getvalue()


class ArchiveInstallTests(unittest.TestCase):
    def _patch_paths(self, temporary):
        root = Path(temporary)
        return (
            mock.patch.object(GENERATOR, "REPO_ROOT", root),
            mock.patch.object(GENERATOR, "BASILISK_DIR", root / "basilisk"),
        )

    def test_archive_rejects_hash_mismatch_before_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            path_patch, basilisk_patch = self._patch_paths(temporary)
            with path_patch, basilisk_patch, mock.patch.object(
                GENERATOR.urllib.request,
                "urlopen",
                return_value=FakeResponse(b"untrusted archive"),
            ), mock.patch.object(GENERATOR.subprocess, "run") as run:
                self.assertFalse(GENERATOR.install_basilisk())
            run.assert_not_called()
            self.assertFalse((Path(temporary) / "basilisk").exists())

    def test_verified_archive_builds_only_literate_c_without_shell(self):
        payload = make_tar_gz(
            {"basilisk/src/darcsit/Makefile": "literate-c:\n\t@true\n"}
        )
        expected_hash = hashlib.sha256(payload).hexdigest()

        def make_literate_c(arguments, **_keywords):
            darcsit = Path(arguments[2])
            executable = darcsit / "literate-c"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            return subprocess.CompletedProcess(arguments, 0)

        with tempfile.TemporaryDirectory() as temporary:
            path_patch, basilisk_patch = self._patch_paths(temporary)
            with path_patch, basilisk_patch, mock.patch.object(
                GENERATOR,
                "BASILISK_ARCHIVES",
                {sys.platform: ("test.tar.gz", expected_hash)},
            ), mock.patch.object(
                GENERATOR.urllib.request,
                "urlopen",
                return_value=FakeResponse(payload),
            ) as open_url, mock.patch.object(
                GENERATOR.subprocess,
                "run",
                side_effect=make_literate_c,
            ) as run:
                self.assertTrue(GENERATOR.install_basilisk())

            request = open_url.call_args.args[0]
            self.assertEqual(
                request.full_url,
                "https://github.com/comphy-lab/basilisk-C/releases/download/"
                f"{GENERATOR.BASILISK_RELEASE}/test.tar.gz",
            )
            arguments, keywords = run.call_args
            self.assertEqual(arguments[0][0], "make")
            self.assertEqual(arguments[0][1], "-C")
            self.assertEqual(arguments[0][-1], "literate-c")
            self.assertFalse(keywords.get("shell", False))
            self.assertTrue(
                (Path(temporary) / "basilisk/src/darcsit/literate-c").is_file()
            )

    def test_archive_rejects_traversal_before_build(self):
        payload = make_tar_gz({"../escape": "nope"})
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            path_patch, basilisk_patch = self._patch_paths(temporary)
            with path_patch, basilisk_patch, mock.patch.object(
                GENERATOR,
                "BASILISK_ARCHIVES",
                {sys.platform: ("test.tar.gz", expected_hash)},
            ), mock.patch.object(
                GENERATOR.urllib.request,
                "urlopen",
                return_value=FakeResponse(payload),
            ), mock.patch.object(GENERATOR.subprocess, "run") as run:
                self.assertFalse(GENERATOR.install_basilisk())
            run.assert_not_called()
            self.assertFalse((Path(temporary) / "escape").exists())

    def test_archive_rejects_links_before_build(self):
        for link_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(link_type=link_type), tempfile.TemporaryDirectory() as temporary:
                payload = make_link_tar_gz(
                    "basilisk/src/darcsit/escape", "../../escape", link_type
                )
                expected_hash = hashlib.sha256(payload).hexdigest()
                path_patch, basilisk_patch = self._patch_paths(temporary)
                with path_patch, basilisk_patch, mock.patch.object(
                    GENERATOR,
                    "BASILISK_ARCHIVES",
                    {sys.platform: ("test.tar.gz", expected_hash)},
                ), mock.patch.object(
                    GENERATOR.urllib.request,
                    "urlopen",
                    return_value=FakeResponse(payload),
                ), mock.patch.object(GENERATOR.subprocess, "run") as run:
                    self.assertFalse(GENERATOR.install_basilisk())
                run.assert_not_called()

    def test_release_archives_are_digest_pinned(self):
        self.assertRegex(GENERATOR.BASILISK_RELEASE, r"^v[0-9]{4}-[0-9]{2}-[0-9]{2}$")
        self.assertEqual(
            GENERATOR.BASILISK_ARCHIVES,
            {
                "linux": (
                    "basilisk-linux.tar.gz",
                    "be22c168121b04f9ad42e1cee960bfc7989870f37e03f2e1e83340fb3ee2e8d3",
                ),
                "darwin": (
                    "basilisk-mac.tar.gz",
                    "5632ff26da923a7029733a990b658e55fbdc11434e118132cfb63e3adf675e95",
                ),
            },
        )
        for archive_name, digest in GENERATOR.BASILISK_ARCHIVES.values():
            self.assertRegex(archive_name, r"^basilisk-(linux|mac)\.tar\.gz$")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class GeneratorExitTests(unittest.TestCase):
    def test_cli_returns_nonzero_when_validation_fails(self):
        with mock.patch.object(GENERATOR, "validate_config", return_value=False):
            self.assertEqual(GENERATOR.cli([]), 1)

    def test_cli_failure_becomes_process_exit_status(self):
        code = textwrap.dedent(
            f"""
            import importlib.util
            import sys
            from pathlib import Path

            path = Path({str(GENERATOR_PATH)!r})
            spec = importlib.util.spec_from_file_location('docs_cli_failure', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            module.validate_config = lambda: False
            raise SystemExit(module.cli([]))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_required_stage_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(GENERATOR, "DOCS_DIR", Path(temporary) / "docs"), \
             mock.patch.object(GENERATOR, "validate_config", return_value=True), \
             mock.patch.object(GENERATOR, "copy_assets", return_value=False):
            self.assertEqual(GENERATOR.cli([]), 1)

    def test_no_sources_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(GENERATOR, "DOCS_DIR", Path(temporary) / "docs"), \
             mock.patch.object(GENERATOR, "validate_config", return_value=True), \
             mock.patch.object(GENERATOR, "copy_assets", return_value=True), \
             mock.patch.object(GENERATOR, "find_source_files", return_value=[]):
            self.assertEqual(GENERATOR.cli([]), 1)

    def test_index_generation_failure_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            with mock.patch.object(GENERATOR, "DOCS_DIR", work / "docs"), \
                 mock.patch.object(GENERATOR, "INDEX_PATH", work / "docs/index.html"), \
                 mock.patch.object(GENERATOR, "BASILISK_DIR", work / "basilisk"), \
                 mock.patch.object(GENERATOR, "validate_config", return_value=True), \
                 mock.patch.object(GENERATOR, "copy_assets", return_value=True), \
                 mock.patch.object(
                     GENERATOR,
                     "find_source_files",
                     return_value=[GENERATOR.REPO_ROOT / "Makefile"],
                 ), \
                 mock.patch.object(
                     GENERATOR,
                     "process_file_with_page2html_logic",
                     return_value=True,
                 ), \
                 mock.patch.object(GENERATOR, "generate_index", return_value=False):
                self.assertEqual(GENERATOR.cli([]), 1)

    def test_conversion_failure_is_aggregated_into_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            with mock.patch.object(GENERATOR, "DOCS_DIR", work / "docs"), \
                 mock.patch.object(GENERATOR, "INDEX_PATH", work / "docs/index.html"), \
                 mock.patch.object(GENERATOR, "BASILISK_DIR", work / "basilisk"), \
                 mock.patch.object(GENERATOR, "validate_config", return_value=True), \
                 mock.patch.object(GENERATOR, "copy_assets", return_value=True), \
                 mock.patch.object(
                     GENERATOR,
                     "find_source_files",
                     return_value=[GENERATOR.REPO_ROOT / "Makefile"],
                 ), \
                 mock.patch.object(
                     GENERATOR,
                     "process_file_with_page2html_logic",
                     return_value=False,
                 ), \
                 mock.patch.object(GENERATOR, "generate_index", return_value=True), \
                 mock.patch.object(
                     GENERATOR, "generate_robots_txt", return_value=True
                 ), \
                 mock.patch.object(GENERATOR, "generate_sitemap", return_value=True):
                self.assertEqual(GENERATOR.cli([]), 1)

    def test_directory_index_preserves_tex_backslashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            docs = work / "docs"
            directory = docs / "src-local"
            directory.mkdir(parents=True)
            generated = directory / "gle-model.h.html"
            generated.write_text(
                '<meta name="description" content="Uses \\mathrm{Ca}">',
                encoding="utf-8",
            )
            template = work / "template.html"
            template.write_text(
                '<html><body><div class="page-content">$body$</div></body></html>',
                encoding="utf-8",
            )
            source = GENERATOR.REPO_ROOT / "src-local" / "gle-model.h"
            with mock.patch.object(GENERATOR, "TEMPLATE_PATH", template):
                self.assertTrue(
                    GENERATOR.generate_directory_index(
                        "src-local",
                        directory,
                        {source: generated},
                        docs,
                        GENERATOR.REPO_ROOT,
                    )
                )
            self.assertIn(
                r"\mathrm{Ca}",
                (directory / "index.html").read_text(encoding="utf-8"),
            )


class WorkflowTests(unittest.TestCase):
    def test_every_external_action_is_commit_pinned(self):
        action_pattern = re.compile(r"uses:\s*[^@\s]+@([0-9a-f]{40})(?:\s|$)")
        floating = []
        for workflow in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
            for line_number, line in enumerate(
                workflow.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "uses:" in line and not action_pattern.search(line):
                    floating.append(f"{workflow.name}:{line_number}: {line.strip()}")
        self.assertEqual(floating, [])

    def test_skip_deploy_marker_matches_literal_marker_only(self):
        helper = REPO_ROOT / ".github" / "scripts" / "has_skip_deploy_marker.sh"

        marked = subprocess.run(
            ["sh", str(helper)],
            input="Refresh search database [skip-deploy]\n",
            text=True,
            check=False,
        )
        self.assertEqual(marked.returncode, 0)

        unmarked = subprocess.run(
            ["sh", str(helper)],
            input="Refresh search database [skipXdeploy]\n",
            text=True,
            check=False,
        )
        self.assertNotEqual(unmarked.returncode, 0)

        deploy_workflow = (
            REPO_ROOT / ".github" / "workflows" / "deploy.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "sh .github/scripts/has_skip_deploy_marker.sh",
            deploy_workflow,
        )

    def test_pages_workflows_build_and_require_index(self):
        for name in ("deploy.yml", "rebuild-on-search-update.yml"):
            workflow = (
                REPO_ROOT / ".github" / "workflows" / name
            ).read_text(encoding="utf-8")
            self.assertIn("bash .github/scripts/build.sh --force-rebuild", workflow)
            self.assertIn("test -s .github/docs/index.html", workflow)
            self.assertIn(
                "pandoc/actions/setup@86321b6dd4675f5014c611e05088e10d4939e09e",
                workflow,
            )
            self.assertIn("version: '2.19.2'", workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
