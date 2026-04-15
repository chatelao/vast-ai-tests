import subprocess
import time
import os
import json
import glob
import sys
import aiohttp
import asyncio

async def wait_for_server(url, timeout=30):
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                async with session.get(f"{url}/v1/models") as response:
                    if response.status == 200:
                        print("Mock server is ready!")
                        return True
            except Exception:
                pass
            print("Waiting for mock server...")
            await asyncio.sleep(2)
    return False

def run_test():
    mock_process = None
    try:
        # Start mock server
        print("Starting mock server...")
        mock_process = subprocess.Popen(
            [sys.executable, "tests/mock_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Wait for server to be ready
        if not asyncio.run(wait_for_server("http://localhost:8000")):
            print("Mock server failed to start within timeout.")
            if mock_process:
                mock_process.terminate()
                stdout, _ = mock_process.communicate()
                print("Mock Server Logs:")
                print(stdout)
            sys.exit(1)

        # Run benchmark
        print("Running benchmark script...")
        orch_command = [
            sys.executable, "benchmark.py",
            "--url", "http://localhost:8000",
            "--model", "tiny-model",
            "--gpu", "micro-test-mock",
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
        result_files = glob.glob("benchmark_micro-test-mock_*.json")
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

        print("Micro mock test PASSED!")

    finally:
        if mock_process:
            print("Stopping mock server...")
            mock_process.terminate()
            try:
                mock_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mock_process.kill()

        # Cleanup result files
        for f in glob.glob("benchmark_micro-test-mock_*.json"):
            os.remove(f)

if __name__ == "__main__":
    run_test()
