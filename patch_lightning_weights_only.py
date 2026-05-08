#!/usr/bin/env python3
"""
Patch for PyTorch Lightning weights_only issue in PyTorch 2.6+
This script monkey-patches the lightning_fabric.utilities.cloud_io._load function
to add weights_only=False parameter, fixing the pathlib.PosixPath error.
"""

import sys
import os

def patch_lightning_weights_only():
    """Patch lightning_fabric to use weights_only=False for torch.load"""
    try:
        from lightning_fabric.utilities.cloud_io import _load as original_load
        import torch
        
        def patched_load(path_or_url: str, map_location=None):
            """Patched version of _load with weights_only=False"""
            if isinstance(path_or_url, (bytes, bytearray)) or hasattr(path_or_url, "read"):
                # any sort of BytesIO or similar
                return torch.load(
                    path_or_url,
                    map_location=map_location,
                    weights_only=False,  # Add weights_only=False
                )
            if str(path_or_url).startswith("http"):
                return torch.hub.load_state_dict_from_url(
                    str(path_or_url),
                    map_location=map_location,
                )
            import fsspec
            from lightning_fabric.utilities.cloud_io import get_filesystem
            fs = get_filesystem(path_or_url)
            with fs.open(path_or_url, "rb") as f:
                return torch.load(f, map_location=map_location, weights_only=False)  # Add weights_only=False
        
        # Apply the patch
        import lightning_fabric.utilities.cloud_io
        lightning_fabric.utilities.cloud_io._load = patched_load
        
        print("✅ Successfully patched lightning_fabric for PyTorch 2.6+ weights_only compatibility")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import lightning_fabric: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to patch lightning_fabric: {e}")
        return False

if __name__ == "__main__":
    success = patch_lightning_weights_only()
    if not success:
        sys.exit(1)
