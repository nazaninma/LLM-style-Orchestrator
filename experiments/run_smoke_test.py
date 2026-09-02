import subprocess
import sys

def main():
    cmd = [sys.executable, "-m", "metamind.main", "--config", "configs/example.yaml"]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    raise SystemExit(r.returncode)

if __name__ == "__main__":
    main()