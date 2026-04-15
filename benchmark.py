import asyncio
import argparse
import sys
import os
from bench.speed_test import run_speed_test_suite
from infra.logging_utils import log_group_start, log_group_end, log_notice, log_error, log_group_cb

async def main():
    parser = argparse.ArgumentParser(description="Run LLM Benchmark Suite")
    parser.add_argument("--gpu", type=str, default="RTX_4090", help="GPU model being tested")
    parser.add_argument("--model", type=str, required=True, help="Model name being tested")
    parser.add_argument("--url", type=str, help="API endpoint URL (defaults to .vast_api_url content)")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 4, 16], help="Concurrency levels")
    parser.add_argument("--requests-per-level", type=int, default=10, help="Requests per level")
    parser.add_argument("--prompt", type=str, default="Explain quantum physics in one sentence.", help="Benchmark prompt")
    parser.add_argument("--wait-timeout", type=int, default=1200, help="Wait timeout in seconds")

    # Email arguments
    parser.add_argument("--email", type=str, help="Recipient email address")
    parser.add_argument("--smtp-host", type=str, default=os.getenv("SMTP_HOST"), help="SMTP host")
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "587")), help="SMTP port")
    parser.add_argument("--smtp-user", type=str, default=os.getenv("SMTP_USER"), help="SMTP user")
    parser.add_argument("--smtp-password", type=str, default=os.getenv("SMTP_PASSWORD"), help="SMTP password")

    args = parser.parse_args()

    api_url = args.url
    if not api_url and os.path.exists(".vast_api_url"):
        with open(".vast_api_url", "r") as f:
            api_url = f.read().strip()

    if not api_url:
        log_error("API URL not provided and .vast_api_url not found.")
        sys.exit(1)

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

    try:
        await run_speed_test_suite(
            gpu_name=args.gpu,
            model_name=args.model,
            api_url=api_url,
            concurrency_levels=args.concurrency_levels,
            requests_per_level=args.requests_per_level,
            prompt=args.prompt,
            email_config=email_config,
            api_key="vllm-benchmark-token",
            log_group_cb=log_group_cb
        )
    except Exception as e:
        log_error(f"Benchmarking failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
