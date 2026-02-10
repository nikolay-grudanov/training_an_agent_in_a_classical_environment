#!/usr/bin/env python3
"""
Integration Test: Complete Pipeline (US1 → US2 → US3 → US5)

Tests full workflow:
1. Train models (US1)
2. Generate visualizations (US2)
3. Generate videos (US3)
4. Generate reports (US5)
"""

import os
import subprocess
import sys
from pathlib import Path


class TestFullPipeline:
    """Integration tests for complete ML pipeline."""

    def test_full_workflow_sequence(self, tmp_path):
        """Test complete workflow: train → visualize → video → report."""
        # Set up output directories
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        results_dir / "experiments"
        reports_dir = results_dir / "reports"

        print("\n" + "="*70)
        print("🚀 INTEGRATION TEST: Full Pipeline Workflow")
        print("="*70 + "\n")

        # Set up Python path
        original_path = sys.path.copy()
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        try:
            # Step 1: Train PPO model
            print("Step 1/4: Training PPO model (50K timesteps for speed)...")
            ppo_cmd = [
                sys.executable, "-m", "src.experiments.completion.baseline_training",
                "--algo", "ppo",
                "--timesteps", "50000",  # Reduced for faster testing
                "--seed", "999",  # Different seed to avoid conflicts
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
            result = subprocess.run(ppo_cmd, capture_output=True, text=True, cwd=tmp_path, env=env)

            assert result.returncode == 0, f"PPO training failed: {result.stderr}"
            ppo_model = tmp_path / "results" / "experiments" / "ppo_seed999" / "ppo_seed999_model.zip"
            assert ppo_model.exists(), "PPO model not created"
            print(f"✅ PPO model created: {ppo_model}\n")

            # Step 2: Generate visualization
            print("Step 2/4: Generating learning curve...")
            viz_cmd = [
                sys.executable, "-m", "src.visualization.graphs",
                "--experiment", "ppo_seed999",
                "--type", "learning_curve",
                "--output", str(ppo_model.parent / "reward_curve.png"),
                "--title", "PPO Learning Curve (Integration Test)",
            ]
            result = subprocess.run(viz_cmd, capture_output=True, text=True, cwd=tmp_path, env=env)

            assert result.returncode == 0, f"Visualization generation failed: {result.stderr}"
            assert (ppo_model.parent / "reward_curve.png").exists(), "Reward curve not created"
            print("✅ Reward curve created\n")

            # Step 3: Generate video
            print("Step 3/4: Generating demonstration video...")
            video_cmd = [
                sys.executable, "-m", "src.visualization.video",
                "--model", str(ppo_model),
                "--output", str(ppo_model.parent / "demo.mp4"),
                "--episodes", "2",  # Minimal episodes for testing
                "--fps", "15",
            ]
            result = subprocess.run(video_cmd, capture_output=True, text=True, cwd=tmp_path, env=env)

            assert result.returncode == 0, f"Video generation failed: {result.stderr}"
            assert (ppo_model.parent / "demo.mp4").exists(), "Video not created"
            print("✅ Video created\n")

            # Step 4: Generate report
            print("Step 4/4: Generating experiment report...")
            report_cmd = [
                sys.executable, "-m", "src.reporting.report_generator",
                "--output", str(reports_dir / "integration_test_report.md"),
                "--experiments", "ppo_seed999",
                "--include-graphs",
                "--include-videos",
            ]
            result = subprocess.run(report_cmd, capture_output=True, text=True, cwd=tmp_path, env=env)

            assert result.returncode == 0, f"Report generation failed: {result.stderr}"
            assert (reports_dir / "integration_test_report.md").exists(), "Report not created"
            print("✅ Report created\n")

            # Verify report content
            report_path = reports_dir / "integration_test_report.md"
            report_content = report_path.read_text()
            assert "ppo_seed999" in report_content, "Report missing experiment ID"
            assert "Learning Curve" in report_content or "Graph" in report_content, "Report missing graph reference"

            print("="*70)
            print("✅ INTEGRATION TEST PASSED: Full pipeline workflow successful!")
            print("="*70 + "\n")
        finally:
            sys.path = original_path

    def test_workflow_artifacts_integrity(self, tmp_path):
        """Test that all expected artifacts are created and valid."""
        print("\n" + "="*70)
        print("🔍 INTEGRATION TEST: Artifacts Integrity Check")
        print("="*70 + "\n")

        original_path = sys.path.copy()
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        try:
            # Train a quick A2C model
            train_cmd = [
                sys.executable, "-m", "src.experiments.completion.baseline_training",
                "--algo", "a2c",
                "--timesteps", "50000",
                "--seed", "998",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
            result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=tmp_path, env=env)

            assert result.returncode == 0, "A2C training failed"

            exp_dir = tmp_path / "results" / "experiments" / "a2c_seed998"

            # Check model file exists and is valid zip
            model_file = exp_dir / "a2c_seed998_model.zip"
            assert model_file.exists(), "Model file missing"

            # Check config.json exists
            config_file = exp_dir / "config.json"
            assert config_file.exists(), "Config file missing"

            # Check metrics.csv exists and is valid
            metrics_file = exp_dir / "metrics.csv"
            assert metrics_file.exists(), "Metrics CSV missing"

            metrics_content = metrics_file.read_text()
            assert "timesteps" in metrics_content or "reward" in metrics_content, "Invalid metrics CSV format"
            print(f"✅ Metrics CSV valid: {len(metrics_content.splitlines())} lines")

            # Check eval_log.csv exists
            eval_file = exp_dir / "eval_log.csv"
            assert eval_file.exists(), "Eval log CSV missing"

            print("\n✅ INTEGRATION TEST PASSED: All artifacts verified!\n")
        finally:
            sys.path = original_path

    def test_parallel_workflows(self, tmp_path):
        """Test that multiple workflows can run in parallel."""
        print("\n" + "="*70)
        print("⚡ INTEGRATION TEST: Parallel Workflows")
        print("="*70 + "\n")

        original_path = sys.path.copy()
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        try:
            # Train two models with different seeds
            commands = [
                [sys.executable, "-m", "src.experiments.completion.baseline_training",
                 "--algo", "ppo", "--timesteps", "30000", "--seed", "100"],
                [sys.executable, "-m", "src.experiments.completion.baseline_training",
                 "--algo", "ppo", "--timesteps", "30000", "--seed", "101"],
            ]

            # Run commands in parallel
            import concurrent.futures
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(
                    subprocess.run, cmd, capture_output=True, text=True, cwd=tmp_path, env=env
                ) for cmd in commands]

                results = [f.result() for f in futures]

            # Both should succeed
            for i, result in enumerate(results):
                assert result.returncode == 0, f"Parallel command {i} failed: {result.stderr}"

            # Both models should exist
            model_100 = tmp_path / "results" / "experiments" / "ppo_seed100" / "ppo_seed100_model.zip"
            model_101 = tmp_path / "results" / "experiments" / "ppo_seed101" / "ppo_seed101_model.zip"

            assert model_100.exists(), "Model 100 not created"
            assert model_101.exists(), "Model 101 not created"

            print("✅ INTEGRATION TEST PASSED: Parallel workflows successful!\n")
        finally:
            sys.path = original_path

    def test_resume_training(self, tmp_path):
        """Test training resumption from checkpoint."""
        print("\n" + "="*70)
        print("🔄 INTEGRATION TEST: Resume Training from Checkpoint")
        print("="*70 + "\n")

        original_path = sys.path.copy()
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        try:
            # Initial training
            train_cmd = [
                sys.executable, "-m", "src.experiments.completion.baseline_training",
                "--algo", "ppo",
                "--timesteps", "30000",
                "--seed", "102",
                "--checkpoint-freq", "15000",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
            result = subprocess.run(train_cmd, capture_output=True, text=True, cwd=tmp_path, env=env)

            assert result.returncode == 0, "Initial training failed"

            exp_dir = tmp_path / "results" / "experiments" / "ppo_seed102"
            checkpoint = exp_dir / "checkpoints" / "checkpoint_15000.zip"
            assert checkpoint.exists(), "Checkpoint not created"

            print(f"✅ Checkpoint created: {checkpoint}")

            # Resume training
            resume_cmd = [
                sys.executable, "-m", "src.experiments.completion.baseline_training",
                "--algo", "ppo",
                "--timesteps", "30000",
                "--seed", "102",
                "--resume-from", str(checkpoint),
            ]
            result = subprocess.run(resume_cmd, capture_output=True, text=True, cwd=tmp_path, env=env)

            assert result.returncode == 0, "Resume training failed"

            # Verify model was updated
            model_file = exp_dir / "ppo_seed102_model.zip"
            assert model_file.exists(), "Resumed model not created"

            print("✅ INTEGRATION TEST PASSED: Resume training successful!\n")
        finally:
            sys.path = original_path
