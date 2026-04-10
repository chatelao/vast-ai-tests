import asyncio
import aiohttp
import time
import json
import argparse
import statistics

class LoadTester:
    def __init__(self, base_url, model_name):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name

    async def send_request(self, session, prompt, max_tokens=100):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True
        }

        start_time = time.perf_counter()
        ttft = None
        tokens_received = 0
        chunk_times = []

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Request failed with status {response.status}: {error_text}")
                    return None

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        if line == "data: [DONE]":
                            break

                        tokens_received += 1
                        current_time = time.perf_counter()

                        if ttft is None:
                            ttft = current_time - start_time
                        else:
                            chunk_times.append(current_time - last_token_time)

                        last_token_time = current_time

            end_time = time.perf_counter()
            total_duration = end_time - start_time

            return {
                "ttft": ttft,
                "itl": statistics.mean(chunk_times) if chunk_times else 0,
                "total_duration": total_duration,
                "tokens": tokens_received,
                "tps": tokens_received / total_duration if total_duration > 0 else 0
            }
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    async def run_benchmark(self, concurrency, num_requests, prompt="Explain quantum physics in one sentence."):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(num_requests):
                tasks.append(self.send_request(session, prompt))

            # Simple concurrency control
            semaphore = asyncio.Semaphore(concurrency)
            async def sem_request(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(*(sem_request(t) for t in tasks))

            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return None

            summary = {
                "concurrency": concurrency,
                "total_requests": len(valid_results),
                "avg_ttft": statistics.mean([r['ttft'] for r in valid_results]),
                "avg_itl": statistics.mean([r['itl'] for r in valid_results]),
                "avg_tps": statistics.mean([r['tps'] for r in valid_results]),
                "total_tps": sum([r['tokens'] for r in valid_results]) / max([r['total_duration'] for r in valid_results])
            }
            return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Load Tester")
    parser.add_argument("--url", type=str, required=True, help="Base URL of the LLM API")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=10, help="Total number of requests")

    args = parser.parse_args()

    tester = LoadTester(args.url, args.model)
    summary = asyncio.run(tester.run_benchmark(args.concurrency, args.requests))
    print(json.dumps(summary, indent=2))
