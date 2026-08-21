import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from semanticsearch.service import SemanticSearchService, chunk_text
from tools import ToolEnvironment, discover_tool_packages


class FakeEmbeddingModel:
    def encode(
        self,
        sentences,
        *,
        batch_size,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        vectors = []
        for sentence in sentences:
            lowered = sentence.lower()
            vector = np.array(
                [lowered.count("ocean"), lowered.count("rocket"), 0.1],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)


def make_model_directory(path: Path) -> None:
    (path / "onnx").mkdir(parents=True)
    for relative in (
        "config.json",
        "modules.json",
        "tokenizer.json",
        "onnx/model.onnx",
        "onnx/model.onnx_data",
    ):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")


class SemanticSearchServiceTest(unittest.TestCase):
    def test_model_is_constructed_only_once_on_first_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            make_model_directory(model_path)
            service = SemanticSearchService(root / "work", root / "indexes", model_path)
            fake = FakeEmbeddingModel()

            with patch("semanticsearch.service.OnnxEmbeddingModel", return_value=fake) as factory:
                first, first_cold, _ = service._get_model()
                second, second_cold, second_seconds = service._get_model()

            factory.assert_called_once_with(model_path.resolve())
            self.assertIs(first, fake)
            self.assertIs(second, fake)
            self.assertTrue(first_cold)
            self.assertFalse(second_cold)
            self.assertEqual(second_seconds, 0.0)

    def test_chunking_returns_bounded_content_and_line_coordinates(self):
        chunks = chunk_text("first line\n" + "ocean " * 100 + "\nlast line", 256, 32)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["content"]) <= 256 for chunk in chunks))
        self.assertEqual(chunks[0]["start_line"], 1)
        self.assertGreaterEqual(chunks[-1]["end_line"], 2)

    def test_builds_mirrored_index_reuses_and_rebuilds_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            model_path = root / "model"
            make_model_directory(model_path)
            source = work / "offline" / "batch" / "article.md"
            source.parent.mkdir(parents=True)
            source.write_text("Ocean currents influence the climate.\n" * 20)
            service = SemanticSearchService(
                work, root / "tmp" / "indexes", model_path, chunk_chars=256
            )
            service._model = FakeEmbeddingModel()

            first = service.search("ocean", "offline", "*.md", 3, -1.0)
            second = service.search("ocean", "offline", "*.md", 3, -1.0)
            source.write_text("Rocket engines produce thrust.\n" * 20)
            third = service.search("rocket", "offline", "*.md", 3, -1.0)

            index_dir = root / "tmp" / "indexes" / "offline" / "batch" / "article.md.index"
            metadata = json.loads((index_dir / "metadata.json").read_text())
            self.assertEqual(first["indexed"], 1)
            self.assertEqual(second["reused"], 1)
            self.assertEqual(third["indexed"], 1)
            self.assertEqual(first["matches"][0]["path"], "offline/batch/article.md")
            self.assertIn("Rocket", third["matches"][0]["content"])
            self.assertEqual(metadata["source_path"], "offline/batch/article.md")
            self.assertTrue((index_dir / "chunks.jsonl").is_file())
            self.assertTrue((index_dir / "embeddings.npy").is_file())

    def test_removes_index_for_deleted_source_inside_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            model_path = root / "model"
            make_model_directory(model_path)
            source = work / "note.txt"
            work.mkdir()
            source.write_text("ocean")
            service = SemanticSearchService(work, root / "indexes", model_path)
            service._model = FakeEmbeddingModel()
            service.search("ocean", ".", None, 3, -1.0)
            source.unlink()

            result = service.search("ocean", ".", None, 3, -1.0)

            self.assertEqual(result["removed"], 1)
            self.assertFalse((root / "indexes" / "note.txt.index").exists())


class SemanticToolAvailabilityTest(unittest.TestCase):
    def test_only_semantic_tool_is_hidden_when_model_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = ToolEnvironment(
                work_dir=root / "work",
                tool_parameters={
                    "files": {"semantic_model_path": str(root / "missing")}
                },
            )
            registry = discover_tool_packages(environment)
            names = {definition.name for definition in registry.definitions()}

            self.assertIn("files.read", names)
            self.assertNotIn("files.semantic_search", names)

    def test_same_namespace_packages_accept_owned_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            make_model_directory(model_path)
            environment = ToolEnvironment(
                work_dir=root / "work",
                tool_parameters={
                    "files": {
                        "semantic_model_path": str(model_path),
                        "semantic_index_root": str(root / "indexes"),
                    }
                },
            )
            registry = discover_tool_packages(environment)

            self.assertIn(
                "files.semantic_search",
                {definition.name for definition in registry.definitions()},
            )
            self.assertIn("conceptual", registry.prompt_instructions())


if __name__ == "__main__":
    unittest.main()
