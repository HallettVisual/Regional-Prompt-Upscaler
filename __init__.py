import subprocess
import sys

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception as e:
        print(f"Failed to install package {package}: {e}")

# Try to import spacy, and install it if it's not available
try:
    import spacy
except ImportError:
    print("Installing 'spacy' package...")
    install('spacy')
