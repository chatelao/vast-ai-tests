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

    async def wait_for_api_ready(self, url, api_key=None, timeout=1200):
        print(f"Waiting for LLM API to be ready at {url}...")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                elapsed = int(time.time() - start_time)
                try:
                    async with session.get(f"{url}/v1/models", headers=headers) as response:
                        if response.status == 200:
                            print(f"API is ready after {elapsed}s!")
                            return True
                        else:
                            text = await response.text()
                            print(f"  ...API returned {response.status}: {text[:100]}")
                except Exception as e:
                    # Generic error (e.g. connection refused) is expected during startup
                    pass
                if elapsed % 30 == 0 and elapsed > 0:
                    print(f"  ...still waiting ({elapsed}s elapsed)")
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

    async def run_suite(self, gpu_name, model_name, url=None, concurrency_levels=[1, 4, 16], requests_per_level=10, wait_timeout=1200, prompt="Explain quantum physics in one sentence.", email_config=None, template_hash="38b2b68cf896e8582dff6f305a2041b1", mode="all"):
        print(f"Starting benchmark suite for {model_name} on {gpu_name} (Mode: {mode})")

        instance_id = None
        vllm_api_key = "vllm-benchmark-token"

        # Load instance ID if it exists (for split-step execution)
        if os.path.exists(".vast_instance_id"):
            with open(".vast_instance_id", "r") as f:
                content = f.read().strip()
                try:
                    instance_id = int(content)
                    print(f"Loaded existing instance ID: {instance_id}")
                except ValueError:
                    # In tests we might use string IDs
                    instance_id = content
                    print(f"Loaded existing instance ID (string): {instance_id}")

        if mode == "teardown":
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
            finally:
                self.log_group_end()
            return

        if url:
            api_url = url
            print(f"Using existing endpoint: {api_url}")
        elif mode == "benchmark":
            if os.path.exists(".vast_api_url"):
                with open(".vast_api_url", "r") as f:
                    api_url = f.read().strip()
                print(f"Loaded API URL from file: {api_url}")
            else:
                raise ValueError("API URL (--url) or .vast_api_url file is required for benchmark mode")
        else:
            if not template_hash:
                raise ValueError("template_hash is required when provisioning a new instance")

            self.log_group_start("Instance Provisioning")
            try:
                # 1. Find and rent instance
                offers = self.vast.find_offers(gpu_name)
                if not offers:
                    self.log_error(f"No offers found for {gpu_name} (or API error occurred)")
                    # Force exit with error code if we can't find offers and were supposed to run
                    raise RuntimeError(f"Could not find any offers for {gpu_name}")

                hf_token = os.getenv("HF_TOKEN", "")
                vllm_args = "--dtype auto --enforce-eager --max-model-len 512 --block-size 16 --port 8000"
                env_vars = f"-e VLLM_MODEL={model_name} -e VLLM_ARGS='{vllm_args}' -e HF_TOKEN={hf_token} -e OPEN_BUTTON_TOKEN={vllm_api_key} -p 1111:1111 -p 7860:7860 -p 8000:8000 -p 8265:8265 -p 8080:8080"

                # Select the best offer (lowest price per hour)
                offer_id = offers[0]['id']
                instance_id = self.vast.rent_instance(offer_id, template_hash=template_hash, env=env_vars)
                if not instance_id:
                    raise RuntimeError(f"Failed to rent instance using offer {offer_id}")

                # Persist instance ID for external cleanup (e.g., GitHub Actions cancellation)
                with open(".vast_instance_id", "w") as f:
                    f.write(str(instance_id))

                # 2. Wait for instance to be ready
                instance = self.vast.wait_for_ssh(instance_id)
                if not instance:
                    raise RuntimeError(f"Instance {instance_id} failed to initialize or become reachable")

                # Resolve external API URL using public IP and port mappings
                host = instance.get('public_ipaddr', instance.get('ssh_host'))
                port = 8000
                ports = instance.get('ports', {})
                for port_key, mappings in ports.items():
                    if port_key.startswith('8000'):
                        if isinstance(mappings, list) and len(mappings) > 0:
                            # Handle standard SDK/Docker port mapping structure
                            mapping = mappings[0]
                            if isinstance(mapping, dict):
                                port = mapping.get('HostPort', port)
                        elif isinstance(mappings, (str, int)):
                            # Handle potential simplified mapping
                            port = mappings
                        break

                api_url = f"http://{host}:{port}"

                with open(".vast_api_url", "w") as f:
                    f.write(api_url)

                print(f"Instance ready at {api_url}")
                print(f"Using template {template_hash}. Waiting for preinstalled vLLM to start...")
                print(f"URL: {api_url}")

                # 2.5 Wait for API to be ready
                if not await self.wait_for_api_ready(api_url, api_key=vllm_api_key, timeout=wait_timeout):
                    raise RuntimeError(f"LLM API at {api_url} never became ready within {wait_timeout} seconds")
            finally:
                self.log_group_end()

            if mode == "provision":
                print("Provisioning complete. API is ready.")
                return

        try:
            if url or mode == "benchmark":
                self.log_group_start("Waiting for API")
                try:
                    # 2.5 Wait for API to be ready
                    if not await self.wait_for_api_ready(api_url, api_key=vllm_api_key, timeout=wait_timeout):
                        raise RuntimeError(f"LLM API at {api_url} never became ready within {wait_timeout} seconds")
                finally:
                    self.log_group_end()

            # 3. Run benchmarks
            tester = LoadTester(api_url, model_name, api_key=vllm_api_key)

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
            if mode == "all":
                self.log_group_start("Teardown")
                try:
                    if instance_id:
                        self.vast.destroy_instance(instance_id)
                        if os.path.exists(".vast_instance_id"):
                            os.remove(".vast_instance_id")
                        if os.path.exists(".vast_api_url"):
                            os.remove(".vast_api_url")
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
