import argparse
import os
from infra.vast_manager import VastManager
from infra.logging_utils import log_group_start, log_group_end, log_notice, log_error

def main():
    parser = argparse.ArgumentParser(description="Teardown Vast.ai Instance")
    parser.add_argument("--instance-id", type=str, help="Specific instance ID to destroy")

    args = parser.parse_args()
    vast = VastManager()

    instance_id = args.instance_id
    if not instance_id and os.path.exists(".vast_instance_id"):
        with open(".vast_instance_id", "r") as f:
            instance_id = f.read().strip()

    if instance_id:
        log_group_start("Infrastructure Teardown")
        log_notice(f"Destroying instance {instance_id}...")
        vast.destroy_instance(instance_id)
        if os.path.exists(".vast_instance_id"):
            os.remove(".vast_instance_id")
        if os.path.exists(".vast_api_url"):
            os.remove(".vast_api_url")
        log_group_end()
    else:
        print("No instance ID found to teardown.")

if __name__ == "__main__":
    main()
