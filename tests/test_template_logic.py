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
    @patch("orchestrator.time.sleep")
    def test_template_hash_passing(self, mock_sleep, MockVastManager):
        # Set default template hash explicitly to avoid mismatch if default changes
        template_hash = "my_template_hash"
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"
        mock_vast.wait_for_ssh.return_value = {
            "public_ipaddr": "1.2.3.4",
            "ssh_host": "ssh4.vast.ai",
            "ssh_port": 2222,
            "ports": {
                "8000/tcp": [{"HostPort": 12345}]
            }
        }

        async def mock_wait(*args, **kwargs):
            return True

        self.orchestrator.wait_for_api_ready = mock_wait

        # Mock LoadTester
        with patch("orchestrator.LoadTester") as MockLoadTester:
            mock_tester = MockLoadTester.return_value

            async def mock_run_bench(concurrency, *args, **kwargs):
                return {
                    "concurrency": concurrency,
                    "avg_ttft": 0.1,
                    "avg_itl": 0.05,
                    "avg_tps": 20.0,
                    "total_tps": 10.0
                }

            mock_tester.run_benchmark = mock_run_bench

            asyncio.run(self.orchestrator.run_suite(
                gpu_name="RTX_4090",
                model_name="gemma",
                template_hash=template_hash,
                concurrency_levels=[1],
                requests_per_level=1
            ))

            # Verify rent_instance was called with template_hash
            vllm_args = "--dtype auto --enforce-eager --max-model-len 512 --block-size 16 --port 8000"
            expected_env = f"-e VLLM_MODEL=gemma -e VLLM_ARGS='{vllm_args}' -e HF_TOKEN= -e OPEN_BUTTON_TOKEN=vllm-benchmark-token -p 1111:1111 -p 7860:7860 -p 8000:8000 -p 8265:8265 -p 8080:8080"
            mock_vast.rent_instance.assert_called_with(123, template_hash=template_hash, env=expected_env)

    @patch("orchestrator.VastManager")
    @patch("orchestrator.time.sleep")
    def test_url_resolution_no_ports(self, mock_sleep, MockVastManager):
        # Verify fallback to port 8000 when ports mapping is missing
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"
        mock_vast.wait_for_ssh.return_value = {
            "public_ipaddr": "1.2.3.4",
            "ssh_host": "ssh4.vast.ai",
            "ssh_port": 2222,
        }

        async def mock_wait(*args, **kwargs):
            return True
        self.orchestrator.wait_for_api_ready = mock_wait
        # Mock get_instance_details to return a realistic response with external port
        mock_vast.get_instance_details.return_value = {
            "public_ipaddr": "1.2.3.4",
            "ports": {
                "8000/tcp": [{"HostPort": 8888}]
            }
        }

        with patch("orchestrator.LoadTester") as MockLoadTester:
            mock_tester = MockLoadTester.return_value
            async def mock_run_bench(concurrency, *args, **kwargs):
                return {"concurrency": concurrency, "total_tps": 10.0}
            mock_tester.run_benchmark = mock_run_bench

            asyncio.run(self.orchestrator.run_suite(
                gpu_name="RTX_4090",
                model_name="gemma-test",
                concurrency_levels=[1],
                requests_per_level=1
            ))

            # Verify LoadTester was initialized with the API key and the correctly resolved URL from get_instance_details
            MockLoadTester.assert_called_with("http://1.2.3.4:8888", "gemma-test", api_key="vllm-benchmark-token")

    @patch("orchestrator.VastManager")
    @patch("orchestrator.time.sleep")
    def test_new_template_env_passing(self, mock_sleep, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"
        mock_vast.wait_for_ssh.return_value = {
            "public_ipaddr": "1.2.3.4",
            "ssh_host": "ssh4.vast.ai",
            "ssh_port": 2222,
            "ports": {
                "8000/tcp": [{"HostPort": 12345}]
            }
        }

        async def mock_wait(*args, **kwargs):
            return True
        self.orchestrator.wait_for_api_ready = mock_wait
        # Mock get_instance_details to return a realistic response with external port
        mock_vast.get_instance_details.return_value = {
            "public_ipaddr": "1.2.3.4",
            "ports": {
                "8000/tcp": [{"HostPort": 12345}]
            }
        }

        with patch("orchestrator.LoadTester") as MockLoadTester:
            mock_tester = MockLoadTester.return_value
            async def mock_run_bench(concurrency, *args, **kwargs):
                return {
                    "concurrency": concurrency,
                    "avg_ttft": 0.1,
                    "avg_itl": 0.05,
                    "avg_tps": 20.0,
                    "total_tps": 10.0
                }
            mock_tester.run_benchmark = mock_run_bench

            with patch.dict(os.environ, {"HF_TOKEN": "test_hf_token"}):
                asyncio.run(self.orchestrator.run_suite(
                    gpu_name="RTX_4090",
                    model_name="gemma-test",
                    template_hash="38b2b68cf896e8582dff6f305a2041b1",
                    concurrency_levels=[1],
                    requests_per_level=1
                ))

            # Verify rent_instance was called with the correct env string
            vllm_args = "--dtype auto --enforce-eager --max-model-len 512 --block-size 16 --port 8000"
            expected_env = f"-e VLLM_MODEL=gemma-test -e VLLM_ARGS='{vllm_args}' -e HF_TOKEN=test_hf_token -e OPEN_BUTTON_TOKEN=vllm-benchmark-token -p 1111:1111 -p 7860:7860 -p 8000:8000 -p 8265:8265 -p 8080:8080"
            mock_vast.rent_instance.assert_called_with(123, template_hash="38b2b68cf896e8582dff6f305a2041b1", env=expected_env)

            # Verify LoadTester was initialized with the API key and the correctly resolved URL
            MockLoadTester.assert_called_with("http://1.2.3.4:12345", "gemma-test", api_key="vllm-benchmark-token")

if __name__ == "__main__":
    unittest.main()
