import asyncio
import json
import time
import argparse
from infra.vast_manager import VastManager
from bench.load_tester import LoadTester

class Orchestrator:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self._vast = None

    @property
    def vast(self):
        if self._vast is None:
            self._vast = VastManager(api_key=self.api_key)
        return self._vast

    async def run_suite(self, gpu_name, model_name, url=None, concurrency_levels=[1, 4, 16], requests_per_level=10, wait_timeout=600):
        print(f"Starting benchmark suite for {model_name} on {gpu_name}")

        instance_id = None
        if url:
            api_url = url
            print(f"Using existing endpoint: {api_url}")
        else:
            # 1. Find and rent instance
            offers = self.vast.find_offers(gpu_name)
            if not offers:
                print(f"No offers found for {gpu_name}")
                return

            # Select the best offer (lowest price per hour)
            offer_id = offers[0]['id']
            instance_id = self.vast.rent_instance(offer_id)
            if not instance_id:
                return

        try:
            if not url:
                # 2. Wait for instance to be ready
                instance = self.vast.wait_for_ssh(instance_id)
                if not instance:
                    print("Instance failed to initialize")
                    return

                print(f"Instance ready at {instance['ssh_host']}:{instance['ssh_port']}")
                api_url = f"http://{instance['ssh_host']}:{instance['ssh_port']}"

                # Implementation Note: In a production environment, this step would involve
                # using an SSH library (like Paramiko) to run 'docker run' on the remote host.
                # Example for vLLM:
                # ssh.exec_command("python -m vllm.entrypoints.openai.api_server --model google/gemma-7b --dtype float --enforce-eager --max-model-len 512 --block-size 16")

                print("To complete the test, ensure the LLM engine is running on the remote host.")
                print(f"URL: {api_url}")

            # 3. Run benchmarks
            tester = LoadTester(api_url, model_name)

            # Wait for server to be ready
            if not await tester.wait_for_ready(timeout=wait_timeout):
                print("Server failed to become ready. Aborting.")
                return

            all_results = []
            for c in concurrency_levels:
                print(f"Running benchmark with concurrency: {c}")
                # We attempt to run the actual load tester
                try:
                    result = await tester.run_benchmark(c, requests_per_level)
                    if result:
                        result["gpu"] = gpu_name
                        result["model"] = model_name
                        all_results.append(result)
                        print(f"Result: {result['total_tps']:.2f} tokens/s")
                    else:
                        print(f"Benchmark failed for concurrency {c} (Is the server running?)")
                except Exception as e:
                    print(f"Error during benchmark: {e}")

            # 4. Save results
            if all_results:
                report_file = f"benchmark_{gpu_name.replace(' ', '_')}_{int(time.time())}.json"
                with open(report_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"Suite complete. Results saved to {report_file}")
            else:
                print("No results collected.")

        finally:
            # 5. Teardown
            if instance_id:
                self.vast.destroy_instance(instance_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Performance Lab Orchestrator")
    parser.add_argument("--gpu", type=str, default="RTX_4090", help="GPU model to test")
    parser.add_argument("--model", type=str, default="gemma-7b", help="Model name")
    parser.add_argument("--url", type=str, help="Existing API endpoint URL (skips provisioning)")
    parser.add_argument("--run", action="store_true", help="Actually run the suite (requires Vast.ai credits)")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 4, 16], help="Concurrency levels to test")
    parser.add_argument("--requests-per-level", type=int, default=10, help="Number of requests per concurrency level")
    parser.add_argument("--wait-timeout", type=int, default=600, help="Seconds to wait for the LLM server to be ready")

    args = parser.parse_args()
    orch = Orchestrator()

    if args.run:
        asyncio.run(orch.run_suite(
            args.gpu,
            args.model,
            url=args.url,
            concurrency_levels=args.concurrency_levels,
            requests_per_level=args.requests_per_level,
            wait_timeout=args.wait_timeout
        ))
    else:
        print("Orchestrator initialized.")
        print(f"Dry-run: Would test {args.model} on {args.gpu}")
        print("Use --run to execute the full pipeline (this will rent a real instance).")
