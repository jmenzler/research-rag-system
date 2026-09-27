from __future__ import annotations

import importlib
import sys
import unittest
from unittest import mock


class OptionalGpuBrokerImportTests(unittest.TestCase):
    def test_mineru_daemon_imports_without_gpu_broker(self) -> None:
        module = importlib.import_module("src.pdf_parsers.mineru_daemon")

        self.assertIsNone(module._BROKER)

    def test_cold_start_fails_before_side_effects_without_gpu_broker(self) -> None:
        module = importlib.import_module("src.pdf_parsers.mineru_daemon")

        with (
            mock.patch.object(module, "_sweep_orphan_mineru"),
            mock.patch.object(module, "pipeline_daemon_url", return_value=None),
            mock.patch.object(module, "_prepare_scratch_root") as prepare_scratch,
        ):
            with self.assertRaisesRegex(RuntimeError, "gpu-broker"):
                module.start_pipeline_daemon()

        prepare_scratch.assert_not_called()

    def test_reranker_daemon_imports_without_gpu_broker(self) -> None:
        sys.modules.pop("scripts.reranker_daemon", None)
        with mock.patch.dict(sys.modules, {"requests": mock.MagicMock()}):
            module = importlib.import_module("scripts.reranker_daemon")

        self.assertIsNone(module._BROKER)

    def test_reranker_spawn_fails_before_side_effects_without_gpu_broker(self) -> None:
        sys.modules.pop("scripts.reranker_daemon", None)
        with mock.patch.dict(sys.modules, {"requests": mock.MagicMock()}):
            module = importlib.import_module("scripts.reranker_daemon")

        manager = module.RerankerManager()
        with (
            mock.patch.object(module.os, "makedirs") as makedirs,
            mock.patch.object(module.subprocess, "Popen") as popen,
        ):
            with self.assertRaisesRegex(RuntimeError, "gpu-broker"):
                manager._spawn()

        makedirs.assert_not_called()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
