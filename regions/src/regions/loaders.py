from pathlib import Path
import numpy as np


def regionDataPath():
    # get path to regions/data/ folder

    # find scr/ directory, data/ must be at same level
    file_path = Path(__file__).resolve()
    parts = file_path.parts
    if 'src' not in parts:
        raise ValueError(f"src/ not found in path")
    idx = parts.index('src')

    return Path(*parts[:idx]) / 'data'