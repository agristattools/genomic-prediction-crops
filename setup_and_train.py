# setup_and_train.py
"""Auto-train script for Streamlit Cloud deployment."""
import subprocess, sys
print("Installing requirements...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
print("\nGenerating training data...")
subprocess.check_call([sys.executable, "src/data_processing/generate_enhanced_data.py"])
print("\nPreprocessing data...")
subprocess.check_call([sys.executable, "src/data_processing/preprocess_enhanced.py"])
print("\nTraining models...")
subprocess.check_call([sys.executable, "src/evaluation/enhanced_predictor.py"])
print("\nSetup complete! Models trained.")