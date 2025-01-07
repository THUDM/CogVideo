import importlib
from pathlib import Path


package_dir = Path(__file__).parent

for subdir in package_dir.iterdir():
    if subdir.is_dir() and not subdir.name.startswith("_"):
        for module_path in subdir.glob("*.py"):
            module_name = module_path.stem
            full_module_name = f".{subdir.name}.{module_name}"
            importlib.import_module(full_module_name, package=__name__)
