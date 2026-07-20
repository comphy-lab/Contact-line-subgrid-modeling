#!/usr/bin/env python3
"""Security and failure-semantics tests for the documentation site."""

import hashlib
import importlib.util
import re
import subprocess
import sys
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


class InstallerTests(unittest.TestCase):
    def test_installer_rejects_hash_mismatch_before_execution(self):
        with mock.patch.object(
            GENERATOR.urllib.request,
            "urlopen",
            return_value=FakeResponse(b"untrusted installer"),
        ), mock.patch.object(GENERATOR.subprocess, "run") as run:
            self.assertFalse(GENERATOR.install_basilisk())
        run.assert_not_called()

    def test_verified_installer_uses_argument_vector_without_shell(self):
        payload = b"#!/bin/bash\nexit 0\n"
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(GENERATOR, "REPO_ROOT", Path(temporary)), \
                 mock.patch.object(
                     GENERATOR, "BASILISK_INSTALLER_SHA256", expected_hash
                 ), \
                 mock.patch.object(
                     GENERATOR.urllib.request,
                     "urlopen",
                     return_value=FakeResponse(payload),
                 ), \
                 mock.patch.object(
                     GENERATOR.subprocess,
                     "run",
                     return_value=subprocess.CompletedProcess([], 0),
                 ) as run:
                self.assertTrue(GENERATOR.install_basilisk())

            arguments, keywords = run.call_args
            self.assertEqual(arguments[0][0], "bash")
            self.assertEqual(
                arguments[0][2:],
                [f"--ref={GENERATOR.BASILISK_RELEASE}", "--hard"],
            )
            self.assertFalse(keywords.get("shell", False))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_installer_source_is_commit_and_digest_pinned(self):
        self.assertRegex(GENERATOR.BASILISK_INSTALLER_COMMIT, r"^[0-9a-f]{40}$")
        self.assertIn(
            f"/{GENERATOR.BASILISK_INSTALLER_COMMIT}/",
            GENERATOR.BASILISK_INSTALLER_URL,
        )
        self.assertRegex(GENERATOR.BASILISK_INSTALLER_SHA256, r"^[0-9a-f]{64}$")


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
