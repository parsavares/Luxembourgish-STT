import subprocess
import sys

def install(package, user=False):
    command = [sys.executable, "-m", "pip", "install"]
    if user:
        command.append("--user")
    command.append(package)
    subprocess.check_call(command)

# Upgrade pip with --user option
try:
    print("Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--upgrade", "pip"])
    print("pip upgraded successfully.")
except subprocess.CalledProcessError as e:
    print(f"Failed to upgrade pip: {e}")
    sys.exit(1)

# List your dependencies here
dependencies = [
    "transformers",
    "torch",
    "gradio",
    "fastapi",
    "uvicorn"
]

for package in dependencies:
    try:
        install(package, user=True)
        print(f"{package} installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        sys.exit(1)

print("All dependencies have been installed successfully.")
