import asyncio
import argparse
import sys
import os
from infra.vast_manager import VastManager
from infra.logging_utils import log_group_start, log_group_end, log_notice, log_error

async def main():
    parser = argparse.ArgumentParser(description="Provision Vast.ai Instance for LLM Benchmarking")
    parser.add_argument("--gpu", type=str, default="RTX_4090", help="GPU model to test")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--template-hash", type=str, default="38b2b68cf896e8582dff6f305a2041b1", help="Vast.ai template hash")
    parser.add_argument("--wait-timeout", type=int, default=1200, help="Wait timeout in seconds")

    args = parser.parse_args()
    vast = VastManager()

    try:
        log_group_start("Infrastructure Provisioning")
        offers = vast.find_offers(args.gpu)
        if not offers:
            raise RuntimeError(f"No offers found for {args.gpu}")

        offer_id = offers[0]['id']
        vllm_api_key = "vllm-benchmark-token"
        env_vars = vast.get_vllm_env_vars(args.model, api_key=vllm_api_key)

        instance_id = vast.rent_instance(offer_id, template_hash=args.template_hash, env=env_vars)
        if not instance_id:
            raise RuntimeError("Failed to rent instance")

        with open(".vast_instance_id", "w") as f:
            f.write(str(instance_id))

        if not vast.wait_for_ssh(instance_id, timeout=args.wait_timeout):
            raise RuntimeError(f"Instance {instance_id} failed to initialize")

        api_url = vast.resolve_api_url(instance_id)
        with open(".vast_api_url", "w") as f:
            f.write(api_url)

        print(f"Waiting for API at {api_url}...")
        if not await vast.wait_for_api_ready(api_url, api_key=vllm_api_key, timeout=args.wait_timeout):
            raise RuntimeError("API failed to become ready")

        log_notice(f"Provisioning successful. API URL: {api_url}")
        log_group_end()
    except Exception as e:
        log_group_end()
        log_error(f"Provisioning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
