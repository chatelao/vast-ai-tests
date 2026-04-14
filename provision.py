import asyncio
import argparse
import sys
from orchestrator import Orchestrator

async def main():
    parser = argparse.ArgumentParser(description="Provision Vast.ai Instance for LLM Benchmarking")
    parser.add_argument("--gpu", type=str, default="RTX_4090", help="GPU model to test")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--template-hash", type=str, default="38b2b68cf896e8582dff6f305a2041b1", help="Vast.ai template hash")
    parser.add_argument("--wait-timeout", type=int, default=1200, help="Wait timeout in seconds")

    args = parser.parse_args()
    orch = Orchestrator()

    try:
        api_url = await orch.provision_instance(
            gpu_name=args.gpu,
            model_name=args.model,
            template_hash=args.template_hash,
            wait_timeout=args.wait_timeout
        )
        print(f"Provisioning successful. API URL: {api_url}")
    except Exception as e:
        orch.log_error(f"Provisioning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
