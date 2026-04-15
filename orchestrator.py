import asyncio
import json
import time
import argparse
import os
from infra.vast_manager import VastManager
from bench.speed_test import run_speed_test_suite, write_step_summary, send_email_report

class Orchestrator:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("VAST_AI_API_KEY")
        self._vast = None

    @property
    def vast(self):
        if self._vast is None:
            self._vast = VastManager(api_key=self.api_key)
        return self._vast

    def log_group_start(self, title):
        if os.getenv("GITHUB_ACTIONS"):
            print(f"::group::{title}")
        else:
            print(f"\n--- {title} ---")

    def log_group_end(self):
        if os.getenv("GITHUB_ACTIONS"):
            print("::endgroup::")

    def log_notice(self, msg):
        if os.getenv("GITHUB_ACTIONS"):
            print(f"::notice::{msg}")
        else:
            print(f"NOTICE: {msg}")

    def log_error(self, msg):
        if os.getenv("GITHUB_ACTIONS"):
            print(f"::error::{msg}")
        else:
            print(f"ERROR: {msg}")

    def write_step_summary(self, results):
        write_step_summary(results)

    async def wait_for_api_ready(self, url, api_key=None, timeout=1200):
        return await self.vast.wait_for_api_ready(url, api_key=api_key, timeout=timeout)

    def send_email_report(self, results, recipient, smtp_config):
        send_email_report(results, recipient, smtp_config)

    def load_instance_id(self):
        if os.path.exists(".vast_instance_id"):
            with open(".vast_instance_id", "r") as f:
                content = f.read().strip()
                try:
                    return int(content)
                except ValueError:
                    return content
        return None

    def load_api_url(self):
        if os.path.exists(".vast_api_url"):
            with open(".vast_api_url", "r") as f:
                return f.read().strip()
        return None

    async def provision_instance(self, gpu_name, model_name, template_hash="38b2b68cf896e8582dff6f305a2041b1", wait_timeout=1200):
        self.log_group_start("Instance Provisioning")
        try:
            offers = self.vast.find_offers(gpu_name)
            if not offers:
                self.log_error(f"No offers found for {gpu_name} (or API error occurred)")
                raise RuntimeError(f"Could not find any offers for {gpu_name}")

            vllm_api_key = "vllm-benchmark-token"
            env_vars = self.vast.get_vllm_env_vars(model_name, api_key=vllm_api_key)

            offer_id = offers[0]['id']
            instance_id = self.vast.rent_instance(offer_id, template_hash=template_hash, env=env_vars)
            if not instance_id:
                raise RuntimeError(f"Failed to rent instance using offer {offer_id}")

            with open(".vast_instance_id", "w") as f:
                f.write(str(instance_id))

            if not self.vast.wait_for_ssh(instance_id, timeout=wait_timeout):
                raise RuntimeError(f"Instance {instance_id} failed to initialize or become reachable")

            api_url = self.vast.resolve_api_url(instance_id)
            self.log_notice(f"API URL: {api_url}")

            with open(".vast_api_url", "w") as f:
                f.write(api_url)

            if not await self.vast.wait_for_api_ready(api_url, api_key=vllm_api_key, timeout=wait_timeout):
                raise RuntimeError(f"LLM API at {api_url} never became ready within {wait_timeout} seconds")

            return api_url
        finally:
            self.log_group_end()

    async def run_benchmark_suite(self, gpu_name, model_name, api_url, concurrency_levels=[1, 4, 16], requests_per_level=10, prompt="Explain quantum physics in one sentence.", email_config=None, wait_timeout=1200):
        vllm_api_key = "vllm-benchmark-token"

        self.log_group_start("Waiting for API")
        try:
            if not await self.vast.wait_for_api_ready(api_url, api_key=vllm_api_key, timeout=wait_timeout):
                raise RuntimeError(f"LLM API at {api_url} never became ready within {wait_timeout} seconds")
        finally:
            self.log_group_end()

        def log_group_cb(title):
            if title: self.log_group_start(title)
            else: self.log_group_end()

        return await run_speed_test_suite(
            gpu_name=gpu_name,
            model_name=model_name,
            api_url=api_url,
            concurrency_levels=concurrency_levels,
            requests_per_level=requests_per_level,
            prompt=prompt,
            email_config=email_config,
            api_key=vllm_api_key,
            log_group_cb=log_group_cb
        )

    def teardown_instance(self):
        instance_id = self.load_instance_id()
        if not instance_id:
            print("No instance ID found to teardown.")
            return

        self.log_group_start("Teardown")
        try:
            self.vast.destroy_instance(instance_id)
            if os.path.exists(".vast_instance_id"):
                os.remove(".vast_instance_id")
            if os.path.exists(".vast_api_url"):
                os.remove(".vast_api_url")
            print(f"Instance {instance_id} destroyed.")
        finally:
            self.log_group_end()

    async def run_suite(self, gpu_name, model_name, url=None, concurrency_levels=[1, 4, 16], requests_per_level=10, wait_timeout=1200, prompt="Explain quantum physics in one sentence.", email_config=None, template_hash="38b2b68cf896e8582dff6f305a2041b1", mode="all"):
        print(f"Starting benchmark suite for {model_name} on {gpu_name} (Mode: {mode})")

        if mode == "teardown":
            self.teardown_instance()
            return

        api_url = url
        if not api_url:
            if mode == "benchmark":
                api_url = self.load_api_url()
                if not api_url:
                    raise ValueError("API URL (--url) or .vast_api_url file is required for benchmark mode")
            else:
                api_url = await self.provision_instance(gpu_name, model_name, template_hash, wait_timeout)

        if mode == "provision":
            print("Provisioning complete. API is ready.")
            return

        try:
            await self.run_benchmark_suite(gpu_name, model_name, api_url, concurrency_levels, requests_per_level, prompt, email_config, wait_timeout)
        finally:
            if mode == "all":
                self.teardown_instance()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Performance Lab Orchestrator")
    parser.add_argument("--gpu", type=str, default="RTX_4090", help="GPU model to test")
    parser.add_argument("--model", type=str, default="google/gemma-2-9b-it", help="Model name")
    parser.add_argument("--url", type=str, help="Existing API endpoint URL (skips provisioning)")
    parser.add_argument("--run", action="store_true", help="Actually run the suite (requires Vast.ai credits)")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 4, 16], help="Concurrency levels to test")
    parser.add_argument("--requests-per-level", type=int, default=10, help="Number of requests per concurrency level")
    parser.add_argument("--wait-timeout", type=int, default=1200, help="Timeout in seconds to wait for API to be ready")
    parser.add_argument("--prompt", type=str, default="Explain quantum physics in one sentence.", help="Prompt to use for benchmarking")
    parser.add_argument("--template-hash", type=str, default="38b2b68cf896e8582dff6f305a2041b1", help="Vast.ai template hash to use for provisioning")
    parser.add_argument("--provision", action="store_true", help="Only provision the instance and wait for API")
    parser.add_argument("--benchmark", action="store_true", help="Only run benchmarks (requires --url or .vast_api_url)")
    parser.add_argument("--teardown", action="store_true", help="Only destroy the instance (requires .vast_instance_id)")

    # Email arguments
    parser.add_argument("--email", type=str, help="Recipient email address for results")
    parser.add_argument("--smtp-host", type=str, default=os.getenv("SMTP_HOST"), help="SMTP server host")
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "587")), help="SMTP server port")
    parser.add_argument("--smtp-user", type=str, default=os.getenv("SMTP_USER"), help="SMTP username")
    parser.add_argument("--smtp-password", type=str, default=os.getenv("SMTP_PASSWORD"), help="SMTP password")

    args = parser.parse_args()
    orch = Orchestrator()

    if args.run:
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

        mode = "all"
        if args.provision: mode = "provision"
        elif args.benchmark: mode = "benchmark"
        elif args.teardown: mode = "teardown"

        try:
            asyncio.run(orch.run_suite(
                args.gpu,
                args.model,
                url=args.url,
                concurrency_levels=args.concurrency_levels,
                requests_per_level=args.requests_per_level,
                wait_timeout=args.wait_timeout,
                prompt=args.prompt,
                email_config=email_config,
                template_hash=args.template_hash,
                mode=mode
            ))
        except Exception as e:
            orch.log_error(f"Error during benchmark run: {e}")
            import sys
            sys.exit(1)
    else:
        print("Orchestrator initialized.")
        print(f"Dry-run: Would test {args.model} on {args.gpu}")
        print("Use --run to execute the full pipeline (this will rent a real instance).")
