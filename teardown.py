import argparse
from orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Teardown Vast.ai Instance")
    # Adding arguments just in case, though current Orchestrator.teardown_instance()
    # uses the persisted .vast_instance_id file.
    parser.add_argument("--instance-id", type=str, help="Specific instance ID to destroy")

    args = parser.parse_args()
    orch = Orchestrator()

    if args.instance_id:
        # If an instance ID is provided, we can manually destroy it.
        # Orchestrator doesn't have a direct method for this yet that doesn't load from file,
        # but we can easily call the manager.
        print(f"Destroying instance {args.instance_id}...")
        orch.vast.destroy_instance(args.instance_id)
    else:
        orch.teardown_instance()

if __name__ == "__main__":
    main()
