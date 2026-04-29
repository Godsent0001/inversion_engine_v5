import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from execution_loop.live_runner import LiveRunner

def main():
    print("========================================")
    print("   LIVE AGENT TRADING ENGINE STARTING   ")
    print("========================================")

    runner = LiveRunner()
    runner.start()

if __name__ == "__main__":
    main()
