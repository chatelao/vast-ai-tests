import unittest
from unittest.mock import MagicMock, patch
import asyncio
import os
from orchestrator import Orchestrator

def run_async(coro):
    return asyncio.run(coro)

class TestTemplateLogic(unittest.TestCase):
    def setUp(self):
        # Ensure VAST_AI_API_KEY is not interfering
        self.env_patcher = patch.dict(os.environ, {"VAST_AI_API_KEY": "fake_key"})
        self.env_patcher.start()
        self.orchestrator = Orchestrator(api_key="test_key")

    def tearDown(self):
        self.env_patcher.stop()

    @patch("orchestrator.VastManager")
    def test_template_hash_passing(self, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"
        mock_vast.wait_for_ssh.return_value = {
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {}
        }

        async def mock_wait(*args, **kwargs):
            return True

        self.orchestrator.wait_for_api_ready = mock_wait

        # Mock LoadTester
        with patch("orchestrator.LoadTester") as MockLoadTester:
            mock_tester = MockLoadTester.return_value

            async def mock_run_bench(*args, **kwargs):
                return {"total_tps": 10.0}

            mock_tester.run_benchmark = mock_run_bench

            asyncio.run(self.orchestrator.run_suite(
                gpu_name="RTX_4090",
                model_name="gemma",
                template_hash="my_template_hash",
                concurrency_levels=[1],
                requests_per_level=1
            ))

            # Verify rent_instance was called with template_hash
            mock_vast.rent_instance.assert_called_with(123, template_hash="my_template_hash", env=None)

    @patch("orchestrator.VastManager")
    def test_new_template_env_passing(self, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"
        mock_vast.wait_for_ssh.return_value = {
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {"8000/tcp": [{"DirectAddress": "mapped.host:32768"}]}
        }

        async def mock_wait(*args, **kwargs):
            return True
        self.orchestrator.wait_for_api_ready = mock_wait

        with patch("orchestrator.LoadTester") as MockLoadTester:
            mock_tester = MockLoadTester.return_value
            async def mock_run_bench(*args, **kwargs):
                return {"total_tps": 10.0}
            mock_tester.run_benchmark = mock_run_bench

            with patch.dict(os.environ, {"HF_TOKEN": "test_hf_token"}):
                asyncio.run(self.orchestrator.run_suite(
                    gpu_name="RTX_4090",
                    model_name="gemma-test",
                    template_hash="7e24e4e5c2e551d012344a9bf4f141c2",
                    concurrency_levels=[1],
                    requests_per_level=1
                ))

            # Verify rent_instance was called with the correct env string
            expected_env = "-e VLLM_MODEL=gemma-test -e VLLM_ARGS='--api-key vllm-benchmark-token --max-model-len 512 --block-size 16 --dtype float --enforce-eager' -e HF_TOKEN=test_hf_token -e OPEN_BUTTON_TOKEN=vllm-benchmark-token -p 8000:18000"
            mock_vast.rent_instance.assert_called_with(123, template_hash="7e24e4e5c2e551d012344a9bf4f141c2", env=expected_env)

            # Verify LoadTester was initialized with the API key
            MockLoadTester.assert_called_with("http://mapped.host:32768", "gemma-test", api_key="vllm-benchmark-token")

    @patch("orchestrator.VastManager")
    def test_api_url_construction_with_mapped_port(self, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"

        # Case 1: Port 8000 is mapped
        mock_vast.wait_for_ssh.return_value = {
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {
                "8000/tcp": [{"DirectAddress": "mapped.host:32768"}]
            }
        }

        async def mock_wait(*args, **kwargs):
            return True
        self.orchestrator.wait_for_api_ready = mock_wait

        with patch("orchestrator.LoadTester") as MockLoadTester:
            mock_tester = MockLoadTester.return_value
            async def mock_run_bench(*args, **kwargs):
                return {"total_tps": 10.0}
            mock_tester.run_benchmark = mock_run_bench

            asyncio.run(self.orchestrator.run_suite(
                gpu_name="RTX_4090",
                model_name="gemma",
                concurrency_levels=[1],
                requests_per_level=1
            ))

            # Check if LoadTester was initialized with the mapped URL
            MockLoadTester.assert_called_with("http://mapped.host:32768", "gemma", api_key=None)

    @patch("orchestrator.VastManager")
    def test_api_url_construction_fallback(self, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"

        # Case 2: Port 8000 is NOT mapped
        mock_vast.wait_for_ssh.return_value = {
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {}
        }

        async def mock_wait(*args, **kwargs):
            return True
        self.orchestrator.wait_for_api_ready = mock_wait

        with patch("orchestrator.LoadTester") as MockLoadTester:
            mock_tester = MockLoadTester.return_value
            async def mock_run_bench(*args, **kwargs):
                return {"total_tps": 10.0}
            mock_tester.run_benchmark = mock_run_bench

            asyncio.run(self.orchestrator.run_suite(
                gpu_name="RTX_4090",
                model_name="gemma",
                concurrency_levels=[1],
                requests_per_level=1
            ))

            # Check if LoadTester was initialized with the SSH URL fallback
            MockLoadTester.assert_called_with("http://1.2.3.4:2222", "gemma", api_key=None)

if __name__ == "__main__":
    unittest.main()
