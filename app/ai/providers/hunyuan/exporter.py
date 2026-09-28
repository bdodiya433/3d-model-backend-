import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.ai.hunyuan.exporter")


class HunyuanExporter:
    """
    Handles exporting trimesh objects produced by Hunyuan3D-2 to standard 3D model formats.
    """

    def export(self, mesh: Any, output_path: Path, format: str = "glb") -> None:
        """
        Exports a generated mesh structure to the specified output path.
        """
        logger.info(f"Exporting generated mesh to: {output_path} (Format: {format})")
        try:
            # If the mesh is packed in a list, extract the first model
            if isinstance(mesh, list):
                if len(mesh) > 0:
                    mesh = mesh[0]
                else:
                    raise ValueError("Mesh generation output contains no valid elements")

            # Check if mesh has export method (standard for trimesh objects)
            if hasattr(mesh, "export"):
                # trimesh.export supports formats like "glb", "obj"
                mesh.export(str(output_path), file_type=format)
                logger.info(f"Mesh successfully written to disk: {output_path.name}")
            else:
                raise TypeError(f"Invalid mesh object type for export: {type(mesh)}")

        except Exception as e:
            logger.error(f"Failed to export 3D mesh: {e}")
            raise e
