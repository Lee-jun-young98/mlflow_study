"""노트북의 실제 학습·MLflow 기록을 임시 로컬 저장소에서 검증한다."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mlflow
import torch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Deep_learning_quick_start.ipynb"


class NotebookEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifact_root = ROOT / ".test-artifacts"
        artifact_root.mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=artifact_root, ignore_cleanup_errors=True)
        cls.addClassCleanup(cls.directory.cleanup)
        folder = Path(cls.directory.name)
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        script = folder / "notebook_under_test.py"
        script.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("notebook_under_test", script)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        cls.addClassCleanup(sys.modules.pop, spec.name, None)

        previous_uri = mlflow.get_tracking_uri()
        cls.addClassCleanup(mlflow.set_tracking_uri, previous_uri)
        mlflow.set_tracking_uri("sqlite:///" + (folder / "tracking.db").as_posix())
        experiment_id = mlflow.create_experiment(
            "notebook-test", artifact_location=(folder / "artifacts").as_uri()
        )
        mlflow.set_experiment(experiment_id=experiment_id)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

        cls.loss_targets = []
        original_forward = torch.nn.CrossEntropyLoss.forward

        def observe_loss(loss, inputs, targets):
            cls.loss_targets.append(targets.detach().cpu().clone())
            return original_forward(loss, inputs, targets)

        # 장치만 CPU로 고정하고 실제 학습·기록·검색 코드를 실행한다.
        with patch("torch.cuda.is_available", return_value=False), patch.object(
            torch.nn.CrossEntropyLoss, "forward", observe_loss
        ):
            spec.loader.exec_module(cls.module)

    def test_disjoint_stratified_split_and_train_only_loss(self):
        m = self.module
        self.assertEqual((len(m.train_df), len(m.test_df)), (120, 30))
        self.assertFalse(set(m.train_df.index) & set(m.test_df.index))
        self.assertEqual(m.test_df["target"].value_counts().to_dict(), {0: 10, 1: 10, 2: 10})
        self.assertEqual(len(self.loss_targets), 101)
        for targets in self.loss_targets:
            torch.testing.assert_close(targets, m.y_train.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(
            m.X_test.cpu(), torch.tensor(m.test_df.iloc[:, :-1].values, dtype=torch.float32)
        )

    def test_all_checkpoints_log_separate_metrics_and_correct_datasets(self):
        models = self.module.ranked_checkpoints
        self.assertEqual(len(models), 11)
        self.assertEqual({model.name for model in models}, {f"torch-iris-{i}" for i in range(0, 101, 10)})
        for model in models:
            metrics = {metric.key: metric for metric in model.metrics}
            self.assertEqual(metrics["test_accuracy"].dataset_name, "test")
            self.assertEqual(metrics["test_accuracy"].dataset_digest, self.module.test_dataset.digest)
            self.assertEqual(metrics["train_accuracy"].dataset_name, "train")
            self.assertEqual(metrics["train_accuracy"].dataset_digest, self.module.train_dataset.digest)

    def test_reload_models_reproduces_test_metrics_and_ranking(self):
        m = self.module
        scores = []
        for model in m.ranked_checkpoints:
            loaded = mlflow.pytorch.load_model(model.model_uri, map_location="cpu")
            with torch.no_grad():
                predicted = loaded(m.X_test.cpu()).argmax(dim=1)
                expected = (predicted == m.y_test.cpu()).float().mean().item()
            recorded = next(metric.value for metric in model.metrics if metric.key == "test_accuracy")
            self.assertAlmostEqual(recorded, expected, places=7)
            scores.append(recorded)
        self.assertEqual(scores, sorted(scores, reverse=True))
        print(f"Verified {len(scores)} checkpoints; highest test accuracy={scores[0]:.4f}")

    def test_accuracy_uses_eval_mode_without_gradients_and_restores_mode(self):
        class Probe(torch.nn.Module):
            def forward(self, inputs):
                self.observed = (self.training, torch.is_grad_enabled())
                return inputs

        model = Probe()
        inputs = torch.tensor([[0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
        labels = torch.tensor([1, 1, 1])
        for initial_mode in (True, False):
            model.train(initial_mode)
            accuracy = self.module.compute_accuracy(model, inputs, labels)
            self.assertAlmostEqual(accuracy, 2 / 3, places=6)
            self.assertEqual(model.observed, (False, False))
            self.assertEqual(model.training, initial_mode)
            self.assertIsNone(inputs.grad)


if __name__ == "__main__":
    unittest.main()
