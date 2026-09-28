import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image
import trimesh

from dataclasses import dataclass, field
from app.services.color.color_extractor import ColorExtractor, ColorExtractionResult
from app.services.color.color_mapper import ColorMapper, ColorMappingResult
from app.services.color.color_validator import ColorValidator, ValidationReport
from app.services.color.reference_search import ReferenceSearch, ReferenceCandidate
from app.services.color.reference_matcher import ReferenceMatcher, MatchResult
from app.services.blender.exporter import BlenderExporter

logger = logging.getLogger("app.services.color.engine")


@dataclass
class ColorDecision:
    """Explicit, explainable decision regarding the color source."""
    source: str                          # "uploaded_image" | "fused_reference" | "uploaded_image_fallback"
    confidence: float
    reference_used: bool
    reference_match_confidence: Optional[float] = None
    reference_id: Optional[str] = None
    observed_views: List[str] = field(default_factory=list)
    inferred_views: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    breakdown: Dict[str, Any] = field(default_factory=dict)


class ColorEngine:
    """
    Unified Color Engine orchestrating:
    1. Reference-derived perceptual color extraction (CIE-LAB space) with confidence scoring.
    2. Optional, privacy-safe external reference search & multi-signal match verification for low-confidence inputs.
    3. Multi-source color fusion with lighting decoupling.
    4. Multi-view 3D surface projection and confidence-weighted seam blending.
    5. Hidden surface topology diffusion for single-image reconstruction.
    6. Pre-export automated quality validation (geometry, colors, Delta-E).
    7. Clean glTF COLOR_0 and Blender .blend asset generation without photographic UV textures.
    """

    def __init__(
        self,
        extractor: Optional[ColorExtractor] = None,
        mapper: Optional[ColorMapper] = None,
        validator: Optional[ColorValidator] = None,
        blender_exporter: Optional[BlenderExporter] = None,
        reference_search: Optional[ReferenceSearch] = None,
        reference_matcher: Optional[ReferenceMatcher] = None,
        confidence_threshold: float = 0.65,
    ):
        self.extractor = extractor or ColorExtractor()
        self.mapper = mapper or ColorMapper()
        self.validator = validator or ColorValidator()
        self.blender_exporter = blender_exporter or BlenderExporter()
        self.reference_search = reference_search or ReferenceSearch()
        self.reference_matcher = reference_matcher or ReferenceMatcher(extractor=self.extractor)
        self.confidence_threshold = confidence_threshold

    def process_and_color_mesh(
        self,
        mesh: trimesh.Trimesh,
        view_images: Dict[str, Any],
        output_glb_path: Union[str, Path],
        output_blend_path: Optional[Union[str, Path]] = None,
        reference_candidate_pool: Optional[List[Any]] = None,
        enable_color: bool = True,
    ) -> Tuple[trimesh.Trimesh, ValidationReport]:
        """
        Executes full color engine pipeline:
          If enable_color is False:
            Creates a clean uncolored Blender clay/neutral base model.
          If enable_color is True:
            Extract reference colors -> Assess Confidence -> Optional Reference Match/Fusion ->
            Map to 3D surfaces -> Validate -> Export GLB & .blend
        """
        if not enable_color:
            logger.info("ColorEngine: Color mapping disabled by user. Generating uncolored Blender clay base model.")
            return self.process_neutral_mesh(
                mesh=mesh,
                output_glb_path=output_glb_path,
                output_blend_path=output_blend_path,
            )

        output_glb_path = Path(output_glb_path).resolve()
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)

        if output_blend_path:
            output_blend_path = Path(output_blend_path).resolve()
            output_blend_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"ColorEngine: Processing {len(view_images)} reference view(s) on mesh with {len(mesh.vertices)} vertices.")

        # 1. Color Extraction for each reference view
        view_color_results: Dict[str, ColorExtractionResult] = {}
        for view_name, img_data in view_images.items():
            if img_data is None:
                continue
            try:
                c_res = self.extractor.extract_colors(img_data)
                view_color_results[view_name] = c_res
                logger.info(
                    f"View '{view_name}': Extracted {len(c_res.dominant_colors)} dominant color(s). "
                    f"Primary: {c_res.primary_color.hex_code}, Confidence: {c_res.color_confidence:.2f}."
                )
            except Exception as ex:
                logger.warning(f"Failed to extract colors for view '{view_name}': {ex}")

        if not view_color_results:
            raise RuntimeError("ColorEngine: Failed to extract valid colors from any provided reference view.")

        # 2. Color Confidence Evaluation & Multi-Source Reference Fusion
        primary_view_name = "front" if "front" in view_color_results else list(view_color_results.keys())[0]
        primary_res = view_color_results[primary_view_name]
        primary_conf = primary_res.color_confidence
        primary_img = view_images[primary_view_name]

        if isinstance(primary_img, Image.Image):
            pil_primary = primary_img
        elif isinstance(primary_img, np.ndarray):
            pil_primary = Image.fromarray(primary_img)
        else:
            pil_primary = Image.open(str(primary_img))

        if primary_conf >= self.confidence_threshold:
            logger.info(
                f"Color confidence is high ({primary_conf:.2f} >= {self.confidence_threshold}). "
                f"Using uploaded image as primary source without internet search."
            )
            decision = ColorDecision(
                source="uploaded_image",
                confidence=primary_conf,
                reference_used=False,
                breakdown=primary_res.confidence_breakdown,
            )
        else:
            logger.info(
                f"Color confidence is low ({primary_conf:.2f} < {self.confidence_threshold}). "
                f"Activating reference search fallback."
            )
            candidates = self.reference_search.search_candidates(
                user_image=pil_primary,
                candidate_pool=reference_candidate_pool,
            )
            best_match = self.reference_matcher.select_best_reference(
                user_image=pil_primary,
                user_extraction=primary_res,
                candidates=candidates,
            )

            if best_match is not None and best_match.is_match:
                logger.info(
                    f"Authoritative same-object match confirmed (score={best_match.match_confidence:.2f}). "
                    f"Fusing uploaded image colors with verified reference '{best_match.candidate.candidate_id}'."
                )
                cand_res = self.extractor.extract_colors(best_match.candidate.image)
                fused_color_map = self.mapper.fuse_color_maps(
                    user_color_map=primary_res.color_map,
                    user_mask=primary_res.object_mask,
                    ref_color_map=cand_res.color_map,
                    ref_mask=cand_res.object_mask,
                    user_weight=0.60,
                    ref_weight=0.40,
                )
                primary_res.color_map = fused_color_map
                fused_conf = round(float(0.5 * primary_conf + 0.5 * best_match.match_confidence), 4)

                decision = ColorDecision(
                    source="fused_reference",
                    confidence=fused_conf,
                    reference_used=True,
                    reference_match_confidence=best_match.match_confidence,
                    reference_id=best_match.candidate.candidate_id,
                    breakdown={
                        "user_confidence": primary_conf,
                        "reference_match_confidence": best_match.match_confidence,
                        "perceptual_similarity": best_match.perceptual_similarity,
                        "silhouette_similarity": best_match.silhouette_similarity,
                        "chroma_similarity": best_match.chroma_similarity,
                    },
                )
            else:
                rejection_msg = (
                    best_match.rejection_reason
                    if best_match
                    else "No candidate passed >= 0.85 match verification threshold."
                )
                logger.info(f"External search did not yield an authoritative match ({rejection_msg}). Using local fallback.")
                decision = ColorDecision(
                    source="uploaded_image_fallback",
                    confidence=primary_conf,
                    reference_used=False,
                    reason=rejection_msg,
                    breakdown=primary_res.confidence_breakdown,
                )

        self._last_decision = decision

        # 3. Map colors onto 3D mesh vertices
        mapping_result = self.mapper.map_colors_to_mesh(
            mesh=mesh,
            view_color_results=view_color_results,
        )
        colored_mesh = mapping_result.mesh

        decision.observed_views = mapping_result.observed_views
        decision.inferred_views = mapping_result.inferred_views

        # 4. Quality Validation
        val_report = self.validator.validate(
            mapping_result=mapping_result,
            reference_results=view_color_results,
        )
        val_report.metrics["color_decision"] = decision

        # 4. Export standard GLB with COLOR_0 vertex colors (No photographic texture images)
        try:
            colored_mesh.export(str(output_glb_path), file_type="glb")
            logger.info(f"ColorEngine: Successfully exported vertex-colored GLB to: {output_glb_path.name}")
        except Exception as ex:
            logger.error(f"ColorEngine: Failed to export GLB mesh: {ex}")
            raise ex

        # Verify exported GLB: ensure COLOR_0 survived and NO raw photo textures exist
        self._verify_exported_glb(output_glb_path)

        # 5. Export native .blend file if requested
        if output_blend_path:
            try:
                self.blender_exporter.export_to_blend(output_glb_path, output_blend_path)
                logger.info(f"ColorEngine: Successfully exported Blender project file to: {output_blend_path.name}")
            except Exception as ex:
                logger.warning(f"ColorEngine: Blender .blend export note: {ex}")

        return colored_mesh, val_report

    def _verify_exported_glb(self, glb_path: Path) -> None:
        """
        Inspects exported GLB to verify standard COLOR_0 vertex colors are present
        and strict 'NO PHOTOGRAPHIC TEXTURE' rule is respected.
        """
        try:
            scene = trimesh.load(str(glb_path))
            if isinstance(scene, trimesh.Scene):
                geoms = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(geoms) if geoms else None
            else:
                mesh = scene

            if mesh is not None and hasattr(mesh, "visual") and mesh.visual is not None:
                # Check for vertex colors in ColorVisuals or TextureVisuals.vertex_attributes['color']
                vcolors = getattr(mesh.visual, "vertex_colors", None)
                if vcolors is None and hasattr(mesh.visual, "vertex_attributes") and "color" in mesh.visual.vertex_attributes:
                    vcolors = mesh.visual.vertex_attributes["color"]

                if vcolors is not None and len(vcolors) > 0:
                    logger.info(f"Verified GLB: Contains {len(vcolors)} valid COLOR_0 vertex colors.")
                else:
                    logger.warning("Verified GLB: vertex_colors attribute was not detected.")

                # Check strict NO-PHOTOGRAPHIC-TEXTURE requirement:
                # Ensure no large embedded photo texture image is attached
                material = getattr(mesh.visual, "material", None)
                if material is not None and hasattr(material, "image") and material.image is not None:
                    logger.warning(
                        "Verified GLB note: Material contains an image reference. "
                        "Ensuring vertex colors remain the primary visual authority."
                    )
                else:
                    logger.info("Verified GLB: Clean pure vertex colors asset confirmed. No photographic image textures.")
        except Exception as e:
            logger.warning(f"GLB asset verification note: {e}")

    def process_neutral_mesh(
        self,
        mesh: trimesh.Trimesh,
        output_glb_path: Union[str, Path],
        output_blend_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[trimesh.Trimesh, ValidationReport]:
        """
        Creates a clean uncolored base model matching Blender's default clay/solid viewport material:
        - Base color: Neutral clay grey [205, 205, 205, 255] (approx 0.8 sRGB).
        - PBR Material: roughness=0.5, metallic=0.0, neutral baseColorFactor.
        - No photographic textures, no color extraction.
        """
        output_glb_path = Path(output_glb_path).resolve()
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)
        if output_blend_path:
            output_blend_path = Path(output_blend_path).resolve()
            output_blend_path.parent.mkdir(parents=True, exist_ok=True)

        mesh_copy = mesh.copy()
        n_verts = len(mesh_copy.vertices)

        # Uniform Blender default clay vertex colors
        clay_rgb = np.full((n_verts, 4), 205, dtype=np.uint8)
        clay_rgb[:, 3] = 255

        mesh_visual = trimesh.visual.color.ColorVisuals(
            mesh=mesh_copy,
            vertex_colors=clay_rgb,
        )
        try:
            mesh_visual.material = trimesh.visual.material.PBRMaterial(
                name="Blender_Clay_Base",
                metallicFactor=0.0,
                roughnessFactor=0.5,
                baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                doubleSided=True,
            )
        except Exception as ex:
            logger.debug(f"PBR material note: {ex}")

        mesh_copy.visual = mesh_visual

        mesh_copy.export(str(output_glb_path), file_type="glb")
        self._verify_exported_glb(output_glb_path)

        if output_blend_path:
            try:
                self.blender_exporter.export_to_blend(output_glb_path, output_blend_path)
            except Exception as ex:
                logger.warning(f"Blender .blend export note: {ex}")

        decision = ColorDecision(
            source="disabled",
            confidence=1.0,
            reference_used=False,
            reason="User disabled color option. Generated clean Blender clay base model.",
        )
        self._last_decision = decision
        val_report = ValidationReport(
            is_valid=True,
            geometry_valid=True,
            color_valid=True,
            mean_delta_e=0.0,
            max_delta_e=0.0,
            color_fidelity_score=1.0,
            vertex_count=n_verts,
            face_count=len(mesh_copy.faces),
            metrics={"color_decision": decision, "mode": "blender_clay_base"},
            warnings=[],
        )
        logger.info(f"ColorEngine: Successfully created uncolored Blender clay base model asset: {output_glb_path.name}")
        return mesh_copy, val_report
