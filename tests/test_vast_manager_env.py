import unittest
from unittest.mock import MagicMock, patch
from infra.vast_manager import VastManager

class TestVastManagerEnv(unittest.TestCase):
    @patch('infra.vast_manager.VastAI')
    def test_rent_instance_converts_env_string_to_dict(self, mock_vastai):
        # Setup
        mock_sdk = MagicMock()
        mock_vastai.return_value = mock_sdk
        mock_sdk.create_instance.return_value = {"success": True, "new_contract": 12345}

        mgr = VastManager(api_key="fake-key")

        # Test with string env
        env_string = "-e VAR1=VAL1 -e VAR2=VAL2 -p 8000:8000"
        mgr.rent_instance(offer_id=111, env=env_string)

        # Verify
        expected_env = {
            "VAR1": "VAL1",
            "VAR2": "VAL2",
            "-p 8000:8000": "1"
        }
        mock_sdk.create_instance.assert_called_with(
            id=111,
            image="nvidia/cuda:12.1.1-devel-ubuntu22.04",
            disk=50,
            template_hash=None,
            env=expected_env
        )

    @patch('infra.vast_manager.VastAI')
    def test_rent_instance_keeps_env_dict(self, mock_vastai):
        # Setup
        mock_sdk = MagicMock()
        mock_vastai.return_value = mock_sdk
        mock_sdk.create_instance.return_value = {"success": True, "new_contract": 12345}

        mgr = VastManager(api_key="fake-key")

        # Test with dict env
        env_dict = {"VAR1": "VAL1"}
        mgr.rent_instance(offer_id=111, env=env_dict)

        # Verify
        mock_sdk.create_instance.assert_called_with(
            id=111,
            image="nvidia/cuda:12.1.1-devel-ubuntu22.04",
            disk=50,
            template_hash=None,
            env=env_dict
        )

if __name__ == '__main__':
    unittest.main()
