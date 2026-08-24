#!/usr/bin/env python3
import json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
def main():
 subprocess.run([sys.executable,str(R/"test_contract.py")],check=True);subprocess.run([sys.executable,"-m","py_compile",str(R/"group_relative_transaction_rl.py"),str(R/"audit_checkpoint.py")],check=True)
 p=json.loads((R/"preregistration.json").read_text());assert p["mandatory_R2"] and p["rollouts_per_prompt"]>=4;print("P8.2 transaction RL preflight: PASS");return 0
if __name__=="__main__":raise SystemExit(main())
