import cv2
import ollama
import pytesseract
import pandas
import PIL
import sys

def verify_setup():
    print("Python Version:", sys.version)
    print("OpenCV Version:", cv2.__version__)
    print("Ollama module imported successfully.")
    print("PyTesseract module imported successfully.")
    print("Pandas Version:", pandas.__version__)
    print("Pillow (PIL) Version:", PIL.__version__)

if __name__ == "__main__":
    try:
        verify_setup()
        print("\nAll dependencies verified successfully!")
    except Exception as e:
        print(f"\nVerification failed: {e}")
        sys.exit(1)
