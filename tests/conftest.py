"""Load the plugin the way Hermes does (a package named after the plugin folder), without Hermes installed."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "live_translate_plugin" not in sys.modules:
    spec = importlib.util.spec_from_file_location("live_translate_plugin", ROOT / "__init__.py",
                                                  submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["live_translate_plugin"] = mod
    spec.loader.exec_module(mod)
