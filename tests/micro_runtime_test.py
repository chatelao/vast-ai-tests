import subprocess
import time
import os
import json
import glob
import sys
import aiohttp
import asyncio

async def wait_for_server(url, timeout=300):
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                async with session.get(f"{url}/v1/models") as response:
                    if response.status == 200:
                        print("Server is ready!")
                        return True
            except Exception:
                pass
            print("Waiting for server...")
            await asyncio.sleep(10)
    return False

def run_test():
    # Environment variables for CPU execution
    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = "cpu"
    env["VLLM_DEVICE"] = "cpu"
    env["VLLM_CPU_KVCACHE_SPACE"] = "1"
    env["VLLM_NO_USAGE_STATS"] = "1"

    vllm_process = None
    try:
        # Start vLLM server
        print("Starting vLLM server with facebook/opt-125m on CPU...")
        vllm_command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "facebook/opt-125m",
            "--dtype", "float",
            "--enforce-eager",
            "--max-model-len", "512",
            "--block-size", "16",
            "--chat-template", "tests/dummy_chat_template.jinja"
        ]

        vllm_process = subprocess.Popen(
            vllm_command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Wait for server to be ready
        if not asyncio.run(wait_for_server("http://localhost:8000")):
            print("vLLM server failed to start within timeout.")
            if vllm_process:
                vllm_process.terminate()
                stdout, _ = vllm_process.communicate()
                print("vLLM Logs:")
                print(stdout)
            sys.exit(1)

        # Run benchmark
        print("Running benchmark script...")
        orch_command = [
            sys.executable, "benchmark.py",
            "--url", "http://localhost:8000",
            "--model", "facebook/opt-125m",
            "--gpu", "micro-test-cpu",
            "--concurrency-levels", "1",
            "--requests-per-level", "2"
        ]

        result = subprocess.run(orch_command, capture_output=True, text=True)
        print("Orchestrator Output:")
        print(result.stdout)
        if result.returncode != 0:
            print("Orchestrator failed:")
            print(result.stderr)
            sys.exit(1)

        # Verify results
        result_files = glob.glob("benchmark_micro-test-cpu_*.json")
        if not result_files:
            print("No benchmark result file found.")
            sys.exit(1)

        latest_file = max(result_files, key=os.path.getctime)
        print(f"Verifying results in {latest_file}...")
        with open(latest_file, "r") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            print("Invalid result data: expected a non-empty list.")
            sys.exit(1)

        metrics = data[0]
        required_keys = ["avg_ttft", "avg_itl", "total_tps"]
        for key in required_keys:
            if key not in metrics or metrics[key] is None:
                print(f"Missing or null metric: {key}")
                sys.exit(1)

        print("Micro runtime test PASSED!")

    finally:
        if vllm_process:
            print("Stopping vLLM server...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                vllm_process.kill()

        # Cleanup result files
        for f in glob.glob("benchmark_micro-test-cpu_*.json"):
            os.remove(f)

if __name__ == "__main__":
    run_test()
