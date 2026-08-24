#!/usr/bin/env python3
"""Local no-GPU preflight for the P8.1.13 protocol."""
import json, subprocess, sys
from pathlib import Path
R=Path(__file__).resolve().parent
def main():
 subprocess.run([sys.executable,str(R/"test_contract.py")],check=True)
 subprocess.run([sys.executable,"-m","py_compile",str(R/"build_preference_pairs.py"),str(R/"train_dpo.py"),str(R/"audit_student.py")],check=True)
 pre=json.loads((R/"preregistration.json").read_text())
 assert pre["mandatory_second_round"] and pre["unified_contract"]["checkpoint"]==1
 print("P8.1.13 preflight: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
