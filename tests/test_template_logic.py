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
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {"18000/tcp": [{"DirectAddress": "mapped.host:32768"}]}
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
            expected_env = "-e VLLM_MODEL=gemma -e HF_TOKEN= -e OPEN_BUTTON_TOKEN=vllm-benchmark-token -p 18000:18000"
            mock_vast.rent_instance.assert_called_with(123, template_hash=template_hash, env=expected_env)

    @patch("orchestrator.VastManager")
    @patch("orchestrator.time.sleep")
    def test_new_template_env_passing(self, mock_sleep, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"
        mock_vast.wait_for_ssh.return_value = {
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {"18000/tcp": [{"DirectAddress": "mapped.host:32768"}]}
        }

        async def mock_wait(*args, **kwargs):
            return True
        self.orchestrator.wait_for_api_ready = mock_wait

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
            expected_env = "-e VLLM_MODEL=gemma-test -e HF_TOKEN=test_hf_token -e OPEN_BUTTON_TOKEN=vllm-benchmark-token -p 18000:18000"
            mock_vast.rent_instance.assert_called_with(123, template_hash="38b2b68cf896e8582dff6f305a2041b1", env=expected_env)

            # Verify LoadTester was initialized with the API key
            MockLoadTester.assert_called_with("http://mapped.host:32768", "gemma-test", api_key="vllm-benchmark-token")

    @patch("orchestrator.VastManager")
    @patch("orchestrator.time.sleep")
    def test_api_url_construction_with_mapped_port(self, mock_sleep, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"

        # Case 1: Port 18000 is mapped
        mock_vast.wait_for_ssh.return_value = {
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {
                "18000/tcp": [{"DirectAddress": "mapped.host:32768"}]
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
            MockLoadTester.assert_called_with("http://mapped.host:32768", "gemma", api_key="vllm-benchmark-token")

    @patch("orchestrator.VastManager")
    @patch("orchestrator.time.sleep")
    @patch("orchestrator.time.time")
    def test_api_url_construction_failure_if_no_port_18000(self, mock_time, mock_sleep, MockVastManager):
        # Mock time to simulate timeout quickly
        # Needs multiple calls: one for start_time, then one for each loop iteration check
        mock_time.side_effect = [100.0, 200.0]

        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"

        # Case 2: Port 18000 is NOT mapped
        mock_vast.wait_for_ssh.return_value = {
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {}
        }
        mock_vast.sdk.show_instances.return_value = [{"id": "inst_1", "ports": {}}]

        with self.assertRaises(RuntimeError) as cm:
            asyncio.run(self.orchestrator.run_suite(
                gpu_name="RTX_4090",
                model_name="gemma",
                concurrency_levels=[1],
                requests_per_level=1
            ))

        self.assertIn("does not have port 18000 mapped", str(cm.exception))

    @patch("orchestrator.VastManager")
    def test_api_url_construction_with_host_ip_port(self, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"

        # HostIp and HostPort are present instead of DirectAddress
        mock_vast.wait_for_ssh.return_value = {
            "id": "inst_1",
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {
                "18000/tcp": [{"HostIp": "5.6.7.8", "HostPort": "32769"}]
            }
        }
        mock_vast.sdk.show_instances.return_value = [
            {
                "id": "inst_1",
                "ports": {"18000/tcp": [{"HostIp": "5.6.7.8", "HostPort": "32769"}]}
            }
        ]

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

            # Check if LoadTester was initialized with the HostIp:HostPort URL
            MockLoadTester.assert_called_with("http://5.6.7.8:32769", "gemma", api_key="vllm-benchmark-token")

    @patch("orchestrator.VastManager")
    @patch("orchestrator.time.sleep") # Speed up tests
    def test_api_url_detection_with_retries(self, mock_sleep, MockVastManager):
        mock_vast = MockVastManager.return_value
        mock_vast.find_offers.return_value = [{"id": 123}]
        mock_vast.rent_instance.return_value = "inst_1"

        # Initially no ports
        instance_no_ports = {
            "id": "inst_1",
            "ssh_host": "1.2.3.4",
            "ssh_port": 2222,
            "ports": {}
        }
        # Later with ports
        instance_with_ports = {
            "id": "inst_1",
            "ports": {"18000/tcp": [{"DirectAddress": "mapped.host:32768"}]}
        }

        mock_vast.wait_for_ssh.return_value = instance_no_ports
        # Mock show_instances to return no ports twice, then with ports
        mock_vast.sdk.show_instances.side_effect = [
            [instance_no_ports],
            [instance_no_ports],
            [instance_with_ports]
        ]

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

            # Should have called show_instances 3 times
            self.assertEqual(mock_vast.sdk.show_instances.call_count, 3)
            # Should have initialized with the eventually discovered URL
            MockLoadTester.assert_called_with("http://mapped.host:32768", "gemma", api_key="vllm-benchmark-token")

if __name__ == "__main__":
    unittest.main()
