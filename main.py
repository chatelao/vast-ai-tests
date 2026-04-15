import asyncio
import argparse
import os
import sys
from infra.vast_manager import VastManager
from bench.speed_test import run_speed_test_suite

def log_group_start(title):
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::group::{title}")
    else:
        print(f"\n--- {title} ---")

def log_group_end():
    if os.getenv("GITHUB_ACTIONS"):
        print("::endgroup::")

def log_notice(msg):
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::notice::{msg}")
    else:
        print(f"NOTICE: {msg}")

def log_error(msg):
    if os.getenv("GITHUB_ACTIONS"):
        print(f"::error::{msg}")
    else:
        print(f"ERROR: {msg}")

def log_group_cb(title):
    if title:
        log_group_start(title)
    else:
        log_group_end()

async def run_end_to_end(args):
    vast = VastManager()
    instance_id = None

    try:
        log_group_start("Infrastructure Provisioning")
        offers = vast.find_offers(args.gpu)
        if not offers:
            log_error(f"No offers found for {args.gpu}")
            return

        offer_id = offers[0]['id']
        vllm_api_key = "vllm-benchmark-token"
        env_vars = vast.get_vllm_env_vars(args.model, api_key=vllm_api_key)

        instance_id = vast.rent_instance(offer_id, template_hash=args.template_hash, env=env_vars)
        if not instance_id:
            log_error("Failed to rent instance")
            return

        if not vast.wait_for_ssh(instance_id):
            log_error("Instance failed to become reachable via SSH")
            return

        api_url = vast.resolve_api_url(instance_id)
        log_notice(f"API URL: {api_url}")

        if not await vast.wait_for_api_ready(api_url, api_key=vllm_api_key):
            log_error("API failed to become ready")
            return
        log_group_end()

        email_config = None
        if args.email:
            email_config = {
                'to': args.email,
                'smtp': {
                    'host': args.smtp_host,
                    'port': args.smtp_port,
                    'user': args.smtp_user,
                    'password': args.smtp_password
                }
            }

        await run_speed_test_suite(
            gpu_name=args.gpu,
            model_name=args.model,
            api_url=api_url,
            concurrency_levels=args.concurrency_levels,
            requests_per_level=args.requests_per_level,
            prompt=args.prompt,
            email_config=email_config,
            api_key=vllm_api_key,
            log_group_cb=log_group_cb
        )

    except Exception as e:
        log_error(f"Unexpected error: {e}")
    finally:
        if instance_id:
            log_group_start("Infrastructure Teardown")
            vast.destroy_instance(instance_id)
            log_group_end()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End LLM Benchmark Orchestrator")
    parser.add_argument("--gpu", type=str, default="RTX_4090", help="GPU model to test")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it", help="Model name")
    parser.add_argument("--template-hash", type=str, default="38b2b68cf896e8582dff6f305a2041b1", help="Vast.ai template hash")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 4, 16], help="Concurrency levels to test")
    parser.add_argument("--requests-per-level", type=int, default=10, help="Number of requests per concurrency level")
    parser.add_argument("--prompt", type=str, default="Explain quantum physics in one sentence.", help="Benchmark prompt")

    # Email arguments
    parser.add_argument("--email", type=str, help="Recipient email address")
    parser.add_argument("--smtp-host", type=str, default=os.getenv("SMTP_HOST"), help="SMTP server host")
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "587")), help="SMTP server port")
    parser.add_argument("--smtp-user", type=str, default=os.getenv("SMTP_USER"), help="SMTP username")
    parser.add_argument("--smtp-password", type=str, default=os.getenv("SMTP_PASSWORD"), help="SMTP password")

    args = parser.parse_args()
    asyncio.run(run_end_to_end(args))
