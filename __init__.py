import subprocess
import sys
import os

# Function to install a package using pip
def install(package):
    try:
        print(f"Installing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Failed to install package {package}: {e}")

# Function to ensure required packages are installed
def ensure_dependencies():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            print("Checking and installing dependencies from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            print("All dependencies are up-to-date.")
        except Exception as e:
            print(f"Error installing dependencies from requirements.txt: {e}")
    else:
        print("requirements.txt not found. Skipping dependency installation.")

# Run the dependency check
ensure_dependencies()
