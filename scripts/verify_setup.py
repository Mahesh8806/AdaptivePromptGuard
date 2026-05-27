# -*- coding: utf-8 -*-
"""
verify_setup.py
Run this script to confirm your environment is correctly set up.
Usage: python scripts/verify_setup.py
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_import(module_name: str, display_name: str = None) -> bool:
    name = display_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "installed")
        print(f"  [OK] {name:<30} {version}")
        return True
    except ImportError as e:
        print(f"  [XX] {name:<30} MISSING -- {e}")
        return False


def check_gpu() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [OK] {'GPU':<30} {gpu}")
            print(f"  [OK] {'VRAM':<30} {vram:.1f} GB")
            if vram >= 8:
                print(f"  [OK] VRAM sufficient for 4-bit quantized 7B models")
            else:
                print(f"  [!!] Low VRAM -- use 4-bit quantization aggressively in Week 2")
            return vram
        else:
            print("  [!!] No CUDA GPU detected -- inference will use CPU (very slow)")
            return 0.0
    except Exception as e:
        print(f"  [XX] GPU check failed: {e}")
        return 0.0


def check_folder_structure() -> bool:
    root = Path(__file__).parent.parent
    required = [
        "data/raw", "data/processed", "data/results",
        "models/llms", "models/classifier",
        "notebooks", "src", "scripts",
        "reports/weekly_logs", "reports/figures", "tests",
    ]
    all_ok = True
    for folder in required:
        p = root / folder
        if p.exists():
            print(f"  [OK] {folder}")
        else:
            print(f"  [XX] {folder}  <-- MISSING")
            all_ok = False
    return all_ok


def check_src_imports() -> bool:
    print("\n[*] Source Module Imports:")
    modules = [
        ("src.config",       "src.config"),
        ("src.utils",        "src.utils"),
        ("src.data_loader",  "src.data_loader"),
        ("src.classifier",   "src.classifier"),
        ("src.prompt_guard", "src.prompt_guard"),
        ("src.pipeline",     "src.pipeline"),
        ("src.evaluator",    "src.evaluator"),
    ]
    results = []
    for mod, name in modules:
        try:
            __import__(mod)
            print(f"  [OK] {name}")
            results.append(True)
        except Exception as e:
            print(f"  [XX] {name:<30} ERROR: {e}")
            results.append(False)
    return all(results)


def main():
    print("\n" + "=" * 55)
    print("   AdaptivePromptGuard (APG) -- Environment Check")
    print("=" * 55)

    print(f"\n[*] Python: {sys.version.split()[0]}  |  Platform: {sys.platform}")

    print("\n[*] Core Libraries:")
    packages = [
        ("torch",        "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets",     "HuggingFace Datasets"),
        ("accelerate",   "Accelerate"),
        ("bitsandbytes", "BitsAndBytes (Quantization)"),
        ("peft",         "PEFT (LoRA)"),
        ("sklearn",      "Scikit-learn"),
        ("pandas",       "Pandas"),
        ("numpy",        "NumPy"),
        ("matplotlib",   "Matplotlib"),
        ("seaborn",      "Seaborn"),
        ("tqdm",         "TQDM"),
        ("loguru",       "Loguru"),
        ("dotenv",       "python-dotenv"),
    ]
    pkg_results = [check_import(pkg, name) for pkg, name in packages]

    print("\n[*] Hardware:")
    vram = check_gpu()

    print("\n[*] Project Folder Structure:")
    folder_ok = check_folder_structure()

    src_ok = check_src_imports()

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    passed = sum(pkg_results)
    print(f"  Packages   : {passed}/{len(pkg_results)} installed")
    print(f"  Folders    : {'OK' if folder_ok else 'Issues found'}")
    print(f"  Src modules: {'OK' if src_ok else 'Issues found'}")
    print(f"  GPU/VRAM   : {vram:.1f} GB" if vram else "  GPU/VRAM   : CPU only")

    if passed == len(pkg_results) and folder_ok and src_ok:
        print("\n  >>> Environment is fully ready for Week 2! <<<")
    else:
        print("\n  [!!] Fix the issues above before proceeding to Week 2.")

    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
