# setup.py
import nltk
import subprocess
import sys

def install_dependencies():
    """Install required Python packages"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    print("NLTK data downloaded successfully.")

if __name__ == "__main__":
    print("Setting up AI Autocorrect Tool...")
    install_dependencies()
    download_nltk_data()
    print("Setup complete! Run 'python app.py' to start the application.")
