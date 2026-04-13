import asyncio
import json
import time
import argparse
import aiohttp
import os
import smtplib
from email.mime.text import MIMEText
from infra.vast_manager import VastManager
from bench.load_tester import LoadTester

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
        summary_file = os.getenv("GITHUB_STEP_SUMMARY")
        if not summary_file or not results:
            return

        try:
            with open(summary_file, "a") as f:
                model = results[0].get('model', 'Unknown')
                gpu = results[0].get('gpu', 'Unknown')
                f.write(f"## Benchmark Results: {model} on {gpu}\n\n")
                f.write("| Concurrency | Avg TTFT (s) | Avg ITL (s) | Avg TPS | Total TPS |\n")
                f.write("|-------------|--------------|-------------|---------|-----------|\n")
                for r in results:
                    c = r.get('concurrency', 'N/A')
                    ttft = r.get('avg_ttft', 0)
                    itl = r.get('avg_itl', 0)
                    avg_tps = r.get('avg_tps', 0)
                    total_tps = r.get('total_tps', 0)
                    f.write(f"| {c} | {ttft:.4f} | {itl:.4f} | {avg_tps:.2f} | {total_tps:.2f} |\n")
                f.write("\n")
        except Exception as e:
            self.log_error(f"Failed to write step summary: {e}")

    async def wait_for_api_ready(self, url, timeout=1200):
        print(f"Waiting for LLM API to be ready at {url}...")
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                try:
                    async with session.get(f"{url}/v1/models") as response:
                        if response.status == 200:
                            print("API is ready!")
                            return True
                except Exception:
                    pass
                await asyncio.sleep(10)
        print("Timeout waiting for API to be ready.")
        return False

    def send_email_report(self, results, recipient, smtp_config):
        """Sends a benchmark report via email."""
        print(f"Sending email report to {recipient}...")
        try:
            body = "LLM Benchmark Results:\n\n"
            body += json.dumps(results, indent=2)

            msg = MIMEText(body)
            msg['Subject'] = f"Benchmark Results: {results[0]['model']} on {results[0]['gpu']}"
            msg['From'] = smtp_config.get('user')
            msg['To'] = recipient

            with smtplib.SMTP(smtp_config.get('host'), smtp_config.get('port')) as server:
                if smtp_config.get('user') and smtp_config.get('password'):
                    server.starttls()
                    server.login(smtp_config.get('user'), smtp_config.get('password'))
                server.send_message(msg)
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email: {e}")

    async def run_suite(self, gpu_name, model_name, url=None, concurrency_levels=[1, 4, 16], requests_per_level=10, wait_timeout=1200, prompt="Explain quantum physics in one sentence.", email_config=None, shutdown_at_exit=False, template_hash=None):
        print(f"Starting benchmark suite for {model_name} on {gpu_name}")

        instance_id = None
        env_vars = None
        vllm_api_key = "vllm-benchmark-token"

        if url:
            api_url = url
            print(f"Using existing endpoint: {api_url}")
        else:
            self.log_group_start("Instance Provisioning")
            try:
                # 1. Find and rent instance
                offers = self.vast.find_offers(gpu_name)
                if not offers:
                    self.log_error(f"No offers found for {gpu_name} (or API error occurred)")
                    # Force exit with error code if we can't find offers and were supposed to run
                    raise RuntimeError(f"Could not find any offers for {gpu_name}")

                # if template_hash == "7e24e4e5c2e551d012344a9bf4f141c2":
                # vllm_args = "--api-key vllm-benchmark-token --max-model-len 512 --block-size 16 --dtype float --enforce-eager"
                # env_vars = f"-e VLLM_MODEL={model_name} -e VLLM_ARGS='{vllm_args}' -e HF_TOKEN={hf_token} -e OPEN_BUTTON_TOKEN={vllm_api_key} -p 8000:18000"

                hf_token = os.getenv("HF_TOKEN", "")
                env_vars = f"-e VLLM_MODEL={model_name} -e HF_TOKEN={hf_token} -e OPEN_BUTTON_TOKEN={vllm_api_key} -p 8000:18000"

                # Select the best offer (lowest price per hour)
                offer_id = offers[0]['id']
                instance_id = self.vast.rent_instance(offer_id, template_hash=template_hash, env=env_vars)
                if not instance_id:
                    raise RuntimeError(f"Failed to rent instance using offer {offer_id}")

                # Persist instance ID for external cleanup (e.g., GitHub Actions cancellation)
                with open(".vast_instance_id", "w") as f:
                    f.write(str(instance_id))
            finally:
                self.log_group_end()

        try:
            if not url:
                self.log_group_start("Waiting for Instance & API")
                try:
                    # 2. Wait for instance to be ready
                    instance = self.vast.wait_for_ssh(instance_id)
                    if not instance:
                        raise RuntimeError(f"Instance {instance_id} failed to initialize or become reachable")

                    # Determine API URL, prioritizing mapped port 8000
                    ports = instance.get('ports', {})
                    if '8000/tcp' in ports:
                        api_url = f"http://{ports['8000/tcp'][0]['DirectAddress']}"
                    else:
                        api_url = f"http://{instance['ssh_host']}:{instance['ssh_port']}"

                    print(f"Instance ready at {api_url}")

                    if template_hash:
                        print(f"Using template {template_hash}. Waiting for preinstalled vLLM to start...")
                    else:
                        print("To complete the test, ensure the LLM engine is running on the remote host.")
                    print(f"URL: {api_url}")

                    # 2.5 Wait for API to be ready
                    if not await self.wait_for_api_ready(api_url, timeout=wait_timeout):
                        raise RuntimeError(f"LLM API at {api_url} never became ready within {wait_timeout} seconds")
                finally:
                    self.log_group_end()
            else:
                self.log_group_start("Waiting for API")
                try:
                    # 2.5 Wait for API to be ready
                    if not await self.wait_for_api_ready(api_url, timeout=wait_timeout):
                        raise RuntimeError(f"LLM API at {api_url} never became ready within {wait_timeout} seconds")
                finally:
                    self.log_group_end()

            # 3. Run benchmarks
            api_key = vllm_api_key if template_hash == "7e24e4e5c2e551d012344a9bf4f141c2" else None
            tester = LoadTester(api_url, model_name, api_key=api_key)

            all_results = []
            for c in concurrency_levels:
                self.log_group_start(f"Benchmark: Concurrency {c}")
                try:
                    print(f"Running benchmark with concurrency: {c}")
                    # We attempt to run the actual load tester
                    try:
                        result = await tester.run_benchmark(c, requests_per_level, prompt=prompt)
                        if result:
                            result["gpu"] = gpu_name
                            result["model"] = model_name
                            all_results.append(result)
                            print(f"Result: {result['total_tps']:.2f} tokens/s")
                        else:
                            print(f"Benchmark failed for concurrency {c} (Is the server running?)")
                    except Exception as e:
                        self.log_error(f"Error during benchmark: {e}")
                finally:
                    self.log_group_end()

            # 4. Save results
            if all_results:
                report_file = f"benchmark_{gpu_name.replace(' ', '_')}_{int(time.time())}.json"
                with open(report_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"Suite complete. Results saved to {report_file}")

                # 4.2 Write GitHub Step Summary
                self.write_step_summary(all_results)

                # 4.5 Send email if configured
                if email_config and email_config.get('to'):
                    self.send_email_report(all_results, email_config['to'], email_config['smtp'])
            else:
                print("No results collected.")

        finally:
            # 5. Teardown
            self.log_group_start("Teardown")
            try:
                if instance_id:
                    self.vast.destroy_instance(instance_id)
                    if os.path.exists(".vast_instance_id"):
                        os.remove(".vast_instance_id")
                elif shutdown_at_exit:
                    # If we are running on the instance itself, try to self-destruct
                    print("Shutdown requested. Attempting to identify current instance ID...")
                    self_id = self.vast.get_current_instance_id()
                    if self_id:
                        self.vast.destroy_instance(self_id)
                    else:
                        print("Could not identify current instance ID for auto-shutdown.")
            finally:
                self.log_group_end()

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
    parser.add_argument("--template-hash", type=str, default="7e24e4e5c2e551d012344a9bf4f141c2", help="Vast.ai template hash to use for provisioning")

    # Email arguments
    parser.add_argument("--email", type=str, help="Recipient email address for results")
    parser.add_argument("--smtp-host", type=str, default=os.getenv("SMTP_HOST"), help="SMTP server host")
    parser.add_argument("--smtp-port", type=int, default=int(os.getenv("SMTP_PORT", "587")), help="SMTP server port")
    parser.add_argument("--smtp-user", type=str, default=os.getenv("SMTP_USER"), help="SMTP username")
    parser.add_argument("--smtp-password", type=str, default=os.getenv("SMTP_PASSWORD"), help="SMTP password")

    # Shutdown argument
    parser.add_argument("--shutdown", action="store_true", help="Auto-shutdown the Vast.ai instance after completion")

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
                shutdown_at_exit=args.shutdown,
                template_hash=args.template_hash
            ))
        except Exception as e:
            orch.log_error(f"Error during benchmark run: {e}")
            import sys
            sys.exit(1)
    else:
        print("Orchestrator initialized.")
        print(f"Dry-run: Would test {args.model} on {args.gpu}")
        print("Use --run to execute the full pipeline (this will rent a real instance).")
