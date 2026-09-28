import sys
from pathlib import Path
import torch


def setup_sys_path() -> None:
    """
    Appends the third-party Hunyuan3D-2 repository folder to the python search path
    to enable direct importing of its packages (hy3dgen, etc.).
    """
    current_file = Path(__file__).resolve()
    # Path levels: app/ai/providers/hunyuan/utils.py -> app/ai/providers/hunyuan -> app/ai/providers -> app/ai -> app -> backend_root
    root_dir = current_file.parents[4]
    third_party_dir = root_dir / "third_party" / "Hunyuan3D-2"
    
    if third_party_dir.exists() and str(third_party_dir) not in sys.path:
        sys.path.insert(0, str(third_party_dir))


def get_torch_device() -> str:
    """
    Resolves the best hardware acceleration device available on the current machine.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_torch_dtype(device: str) -> torch.dtype:
    """
    Returns the appropriate tensor floating point representation for the given device.
    """
    if device == "cpu":
        return torch.float32
    return torch.float16
