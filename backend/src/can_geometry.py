"""
can_geometry.py - Compatibility wrapper for 3D Geometry Reconstruction.

Redirects to GeometryReconstructor for true image-driven multi-view reconstruction.
No static/predefined templates or product-specific dimensions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.geometry_reconstruction import GeometryReconstructor

logger = logging.getLogger("backend.can_geometry")


class CanGeometryBuilder:
    """
    Backward-compatible wrapper redirecting to GeometryReconstructor.
    Reconstructs 3D geometry from images rather than using static primitives.
    """

    def __init__(self, blender_bin: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
        self.reconstructor = GeometryReconstructor(blender_bin=blender_bin)

    def build_geometry(
        self,
        segmented_views: Dict[str, Dict[str, Any]],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Reconstructs image-driven 3D geometry from segmented views.
        """
        return self.reconstructor.reconstruct_from_views(segmented_views, output_dir)

    def build_can(
        self,
        output_dir: Path,
        diameter: float = 0.66,
        height: float = 1.15,
        n_segments: int = 64,
        segmented_views: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Path]:
        """
        Backward-compatibility method.
        If segmented_views are provided, uses image-driven reconstruction.
        """
        if segmented_views:
            return self.build_geometry(segmented_views, output_dir)
        
        # Fallback only when no views provided
        logger.warning("No segmented views provided to build_can; building procedural mesh.")
        return self.reconstructor.reconstruct_from_views({}, output_dir)
