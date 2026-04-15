import asyncio
import aiohttp
import time
import json
import statistics
import os
import smtplib
from email.mime.text import MIMEText

class LoadTester:
    def __init__(self, base_url, model_name, api_key=None):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key

    async def send_request(self, session, prompt, max_tokens=100):
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

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
            async with session.post(url, json=payload, headers=headers) as response:
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
            completed = 0
            total = len(tasks)

            async def sem_request(task):
                nonlocal completed
                async with semaphore:
                    res = await task
                    completed += 1
                    if completed % 5 == 0 or completed == total:
                        print(f"  Progress: {completed}/{total} requests finished")
                    return res

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

def write_step_summary(results):
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
        print(f"Failed to write step summary: {e}")

def send_email_report(results, recipient, smtp_config):
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

async def run_speed_test_suite(gpu_name, model_name, api_url, concurrency_levels=[1, 4, 16], requests_per_level=10, prompt="Explain quantum physics in one sentence.", email_config=None, api_key="vllm-benchmark-token", log_group_cb=None):
    tester = LoadTester(api_url, model_name, api_key=api_key)
    all_results = []

    for c in concurrency_levels:
        if log_group_cb: log_group_cb(f"Benchmark: Concurrency {c}")
        else: print(f"\n--- Benchmark: Concurrency {c} ---")

        try:
            print(f"Running benchmark with concurrency: {c}")
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
                print(f"Error during benchmark: {e}")
        finally:
            if log_group_cb: log_group_cb(None) # End group

    if all_results:
        report_file = f"benchmark_{gpu_name.replace(' ', '_')}_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Suite complete. Results saved to {report_file}")
        write_step_summary(all_results)
        if email_config and email_config.get('to'):
            send_email_report(all_results, email_config['to'], email_config['smtp'])
    else:
        print("No results collected.")

    return all_results
