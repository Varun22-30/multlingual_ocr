# env_doctor.py
# A script to diagnose the Python environment and paths.

import sys
import os

print("--- Python Environment Report ---")

print(f"\n1. Python Version:\n   {sys.version}")
print(f"\n2. Python Executable Path:\n   {sys.executable}")
print(f"\n3. Current Working Directory:\n   {os.getcwd()}")

print("\n4. Python's Search Paths (sys.path):")
for i, path in enumerate(sys.path):
    print(f"   - {path}")

print("\n5. Checking for 'src/__init__.py' from this script's perspective:")
init_path = os.path.join(os.getcwd(), 'src', '__init__.py')
if os.path.exists(init_path):
    print(f"   ✅ SUCCESS: Found the file at '{init_path}'")
else:
    print(f"   ❌ CRITICAL ERROR: Could NOT find the file at '{init_path}'")
    
print("\n--- Report Complete ---")