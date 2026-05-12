import os
import sys

# Add the inversion_engine_v5 directory to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(script_dir, "inversion_engine_v5")
sys.path.append(engine_path)

from inversion_engine_v5.research.experiments.run_full_sim import main

if __name__ == "__main__":
    main()
