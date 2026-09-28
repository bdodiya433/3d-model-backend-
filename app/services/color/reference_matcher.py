import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from PIL import Image
import cv2

from app.services.color.reference_search import ReferenceCandidate
from app.services.color.color_extractor import ColorExtractor, ColorExtractionResult

logger = logging.getLogger("app.services.color.reference_matcher")


@dataclass
class MatchResult:
    """Detailed result of verifying an external reference candidate against the user's uploaded object."""
    candidate: ReferenceCandidate
    is_match: bool
    match_confidence: float                 # Overall match score [0.0, 1.0]
    perceptual_similarity: float           # dHash perceptual similarity [0.0, 1.0]
    silhouette_similarity: float           # Object mask IoU & aspect similarity [0.0, 1.0]
    spatial_region_match: bool             # Strict Section 11: Top/Middle/Bottom spatial alignment
    chroma_similarity: float               # Lighting-decoupled LAB chromaticity match [0.0, 1.0]
    rejection_reason: Optional[str] = None
    signal_breakdown: Dict[str, Any] = field(default_factory=dict)


class ReferenceMatcher:
    """
    Evaluates whether an external candidate image represents the EXACT SAME OBJECT
    as the user's uploaded image.
    
    Principles:
    1. Do NOT Trust Internet Results Blindly:
       Candidates are rejected unless they cross a strict multi-signal confidence threshold (>= 0.85).
    2. Lighting Decoupling (Section 10):
       Separates object base chromaticity (a*, b*) from ambient lighting/shadow (L*).
    3. Spatial Color Region Verification (Section 11):
       Verifies spatial layout (e.g. Top=Red, Middle=Blue, Bottom=Yellow). Rejects inverted or
       scrambled color layouts even if global palette matches.
    4. Silhouette & Geometry Alignment:
       Compares segmented object aspect ratio, shape contours, and solidity.
    """

    def __init__(
        self,
        match_threshold: float = 0.85,
        extractor: Optional[ColorExtractor] = None,
    ):
        self.match_threshold = match_threshold
        self.extractor = extractor or ColorExtractor()

    def _compute_dhash(self, image: Image.Image, hash_size: int = 8) -> np.ndarray:
        """Computes difference hash (dHash) for fast perceptual image comparison."""
        resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        arr = np.array(resized, dtype=np.float32)
        # Compare adjacent pixels horizontally
        diff = arr[:, 1:] > arr[:, :-1]
        return diff.flatten()

    def compute_perceptual_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculates normalized perceptual similarity [0.0, 1.0] using 64-bit dHash Hamming distance.
        """
        hash1 = self._compute_dhash(img1)
        hash2 = self._compute_dhash(img2)
        hamming_dist = np.count_nonzero(hash1 != hash2)
        sim = 1.0 - (hamming_dist / float(len(hash1)))
        return float(np.clip(sim, 0.0, 1.0))

    def evaluate_spatial_color_regions(
        self,
        user_rgba: np.ndarray,
        user_mask: np.ndarray,
        cand_rgba: np.ndarray,
        cand_mask: np.ndarray,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Strict Section 11 Verification:
        Divides object bounding box into 3 vertical zones (Top, Middle, Bottom).
        Verifies that chromaticity (a*, b*) matches in each spatial zone.
        Rejects candidate if colors are vertically swapped or inverted.
        """
        def get_zone_chroma(rgba: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
            coords = np.argwhere(mask)
            if len(coords) < 30:
                return None
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            h = max(1, ymax - ymin)

            rgb_f = rgba[..., :3].astype(np.float32) / 255.0
            lab_f = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2LAB)

            zones = {}
            # Divide into Top (0-33%), Mid (33-66%), Bot (66-100%)
            y_slices = [
                ("top", ymin, ymin + h // 3),
                ("mid", ymin + h // 3, ymin + 2 * h // 3),
                ("bot", ymin + 2 * h // 3, ymax + 1),
            ]
            for name, y0, y1 in y_slices:
                zone_mask = np.zeros_like(mask, dtype=bool)
                zone_mask[y0:y1, :] = mask[y0:y1, :]
                if np.count_nonzero(zone_mask) > 10:
                    zone_pixels = lab_f[zone_mask]
                    # Median a*, b* (chromaticity, ignoring L* lighting differences)
                    median_ab = np.median(zone_pixels[:, 1:3], axis=0)
                    zones[name] = median_ab
                else:
                    zones[name] = np.array([128.0, 128.0])
            return zones

        user_zones = get_zone_chroma(user_rgba, user_mask)
        cand_zones = get_zone_chroma(cand_rgba, cand_mask)

        if not user_zones or not cand_zones:
            return False, 0.0, {"error": "insufficient_zone_pixels"}

        zone_diffs = {}
        max_zone_diff = 0.0
        for zone in ("top", "mid", "bot"):
            u_ab = user_zones[zone]
            c_ab = cand_zones[zone]
            # Euclidean distance in CIE-LAB chromaticity plane (Delta E_ab)
            delta_e_chroma = float(np.linalg.norm(u_ab - c_ab))
            zone_diffs[zone] = round(delta_e_chroma, 2)
            if delta_e_chroma > max_zone_diff:
                max_zone_diff = delta_e_chroma

        # If any zone has delta_e > 32 (e.g. Red vs Blue is delta_e > 60), reject!
        spatial_match = (max_zone_diff <= 32.0)
        mean_diff = float(np.mean(list(zone_diffs.values())))
        # Convert mean delta_e to similarity [0.0, 1.0] (delta_e=0 -> 1.0, delta_e=35 -> 0.0)
        chroma_sim = float(np.clip(1.0 - (mean_diff / 35.0), 0.0, 1.0))

        details = {
            "zone_chroma_diffs": zone_diffs,
            "max_zone_diff": round(max_zone_diff, 2),
            "mean_chroma_diff": round(mean_diff, 2),
            "spatial_match": spatial_match,
        }
        return spatial_match, chroma_sim, details

    def compute_silhouette_similarity(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Computes silhouette similarity between two segmented masks using IoU and aspect ratio.
        """
        # Crop both masks to tight bounding boxes
        def crop_mask(m: np.ndarray) -> np.ndarray:
            coords = np.argwhere(m > 0)
            if len(coords) < 10:
                return np.zeros((64, 64), dtype=np.uint8)
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            return m[ymin:ymax+1, xmin:xmax+1]

        c1 = crop_mask(mask1)
        c2 = crop_mask(mask2)

        ar1 = c1.shape[1] / float(max(1, c1.shape[0]))
        ar2 = c2.shape[1] / float(max(1, c2.shape[0]))
        ar_sim = float(np.clip(1.0 - abs(ar1 - ar2) / max(ar1, ar2, 1e-4), 0.0, 1.0))

        # Resize to common 64x64 canvas to compute IoU
        r1 = cv2.resize(c1.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST) > 0
        r2 = cv2.resize(c2.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST) > 0

        intersection = np.logical_and(r1, r2).sum()
        union = np.logical_or(r1, r2).sum()
        iou = float(intersection) / float(max(1, union))

        return float(0.6 * iou + 0.4 * ar_sim)

    def verify_candidate(
        self,
        user_image: Image.Image,
        user_extraction: ColorExtractionResult,
        candidate: ReferenceCandidate,
    ) -> MatchResult:
        """
        Runs comprehensive multi-signal verification on a candidate reference image.
        Returns MatchResult indicating whether candidate represents the same object.
        """
        # 1. Perceptual Similarity
        perceptual_sim = self.compute_perceptual_similarity(user_image, candidate.image)

        # 2. Extract candidate colors & segmentation mask
        cand_rgba = np.array(candidate.image.convert("RGBA"))
        cand_mask = self.extractor.compute_object_mask(cand_rgba)

        user_rgba = np.array(user_image.convert("RGBA"))
        user_mask = user_extraction.object_mask

        # 3. Silhouette Similarity
        sil_sim = self.compute_silhouette_similarity(user_mask, cand_mask)

        # 4. Strict Spatial Color Region Alignment (Section 11)
        spatial_match, chroma_sim, spatial_details = self.evaluate_spatial_color_regions(
            user_rgba=user_rgba,
            user_mask=user_mask,
            cand_rgba=cand_rgba,
            cand_mask=cand_mask,
        )

        # 5. Multi-Signal Aggregate Confidence Score
        # Weighted blend: 35% spatial chromaticity, 35% silhouette, 30% perceptual
        weights = {"chroma": 0.35, "silhouette": 0.35, "perceptual": 0.30}
        match_confidence = (
            weights["chroma"] * chroma_sim
            + weights["silhouette"] * sil_sim
            + weights["perceptual"] * perceptual_sim
        )

        rejection_reason = None
        # Strict rules:
        # Rule A: If spatial color regions are mismatched (e.g. colors vertically inverted), MUST REJECT!
        if not spatial_match:
            is_match = False
            rejection_reason = f"Spatial color region mismatch: {spatial_details.get('max_zone_diff')} Delta-E."
            match_confidence = min(match_confidence, 0.50)
        # Rule B: Overall score must cross threshold
        elif match_confidence < self.match_threshold:
            is_match = False
            rejection_reason = f"Match confidence ({match_confidence:.2f}) below threshold ({self.match_threshold})."
        else:
            is_match = True

        signal_breakdown = {
            "match_confidence": round(match_confidence, 4),
            "perceptual_similarity": round(perceptual_sim, 4),
            "silhouette_similarity": round(sil_sim, 4),
            "chroma_similarity": round(chroma_sim, 4),
            "spatial_region_match": spatial_match,
            "spatial_details": spatial_details,
        }

        logger.info(
            f"Candidate '{candidate.candidate_id}' verification: "
            f"is_match={is_match}, confidence={match_confidence:.2f}, "
            f"spatial_match={spatial_match}, reason={rejection_reason}."
        )

        return MatchResult(
            candidate=candidate,
            is_match=is_match,
            match_confidence=round(match_confidence, 4),
            perceptual_similarity=round(perceptual_sim, 4),
            silhouette_similarity=round(sil_sim, 4),
            spatial_region_match=spatial_match,
            chroma_similarity=round(chroma_sim, 4),
            rejection_reason=rejection_reason,
            signal_breakdown=signal_breakdown,
        )

    def select_best_reference(
        self,
        user_image: Image.Image,
        user_extraction: ColorExtractionResult,
        candidates: List[ReferenceCandidate],
    ) -> Optional[MatchResult]:
        """
        Evaluates all candidates and selects the single best verified match crossing threshold.
        Returns None if all candidates are rejected.
        """
        if not candidates:
            return None

        verified_matches: List[MatchResult] = []
        for cand in candidates:
            res = self.verify_candidate(user_image, user_extraction, cand)
            if res.is_match:
                verified_matches.append(res)

        if not verified_matches:
            return None

        # Sort by match confidence descending
        verified_matches.sort(key=lambda m: m.match_confidence, reverse=True)
        best = verified_matches[0]
        logger.info(f"Selected authoritative reference match '{best.candidate.candidate_id}' with score {best.match_confidence:.2f}.")
        return best
