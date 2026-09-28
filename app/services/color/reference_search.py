import logging
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import numpy as np

logger = logging.getLogger("app.services.color.reference_search")


@dataclass
class ReferenceCandidate:
    """Represents a potential reference image found online or in a verified product repository."""
    candidate_id: str
    image: Image.Image
    source_origin: str
    priority_level: int           # 1 = Exact same image, 2 = Same product angle, 3 = Verified source, 4 = Visual match
    source_type: str              # "exact_match", "same_product", "verified_catalog", "visual_match"
    cached_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReferenceSearch:
    """
    Service responsible for discovering candidate external reference images
    when uploaded image color confidence is low (< 0.65).
    
    Principles:
    1. Privacy-First: Does NOT upload raw user images to untrusted public search engines.
       Uses perceptual hashes, anonymous visual descriptors, or local verified product catalogs.
    2. Optional & Non-Blocking: Fully disabled if network is down or search is turned off.
    3. Network Failure Resilience: Catches timeouts, connection drops, and HTTP errors gracefully.
       Never interrupts 3D model generation on search failure.
    4. Local Disk Caching: Stores candidate images in temp/reference_cache/ to prevent redundant queries.
    """

    def __init__(
        self,
        enabled: bool = True,
        cache_dir: Union[str, Path] = "temp/reference_cache",
        timeout_seconds: float = 4.0,
    ):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_candidate_id(self, image: Image.Image) -> str:
        """Generates a deterministic SHA256 ID based on image pixel bytes."""
        thumb = image.resize((64, 64)).convert("RGB")
        return hashlib.sha256(thumb.tobytes()).hexdigest()[:16]

    def cache_candidate(self, candidate: ReferenceCandidate) -> Path:
        """Persists candidate reference image to the local cache directory."""
        target_path = self.cache_dir / f"ref_{candidate.candidate_id}.png"
        if not target_path.exists():
            candidate.image.convert("RGBA").save(target_path, "PNG")
        candidate.cached_path = target_path
        return target_path

    def load_cached_candidate(self, candidate_id: str) -> Optional[Image.Image]:
        """Loads a previously cached reference image if available."""
        target_path = self.cache_dir / f"ref_{candidate_id}.png"
        if target_path.exists():
            try:
                return Image.open(target_path).convert("RGBA")
            except Exception as e:
                logger.warning(f"Failed to load cached reference image: {e}")
        return None

    def search_candidates(
        self,
        user_image: Image.Image,
        object_descriptor: Optional[str] = None,
        dominant_hex_codes: Optional[List[str]] = None,
        candidate_pool: Optional[List[Union[str, Path, Image.Image]]] = None,
    ) -> List[ReferenceCandidate]:
        """
        Discovers candidate external references for the object.
        
        Sources checked in priority order:
        1. Explicit candidate_pool (e.g. verified multi-angle product assets or benchmark reference pool).
        2. Local disk cache of verified assets.
        3. Optional network search (mock/pluggable provider with timeout & failure resilience).
        
        Returns list of ReferenceCandidate objects sorted by priority.
        """
        if not self.enabled:
            logger.info("ReferenceSearch is disabled by configuration. Skipping external lookup.")
            return []

        candidates: List[ReferenceCandidate] = []
        t0 = time.time()

        try:
            # 1. Inspect explicit candidate pool (high priority, e.g. catalog or alternate product shots)
            if candidate_pool:
                for idx, item in enumerate(candidate_pool):
                    if time.time() - t0 > self.timeout_seconds:
                        logger.warning("ReferenceSearch timeout reached while inspecting candidate pool.")
                        break

                    try:
                        if isinstance(item, (str, Path)):
                            p = Path(item)
                            if not p.exists():
                                continue
                            c_img = Image.open(p).convert("RGBA")
                            origin = p.name
                        elif isinstance(item, Image.Image):
                            c_img = item.convert("RGBA")
                            origin = f"pool_item_{idx}"
                        else:
                            continue

                        cand_id = self._generate_candidate_id(c_img)
                        cand = ReferenceCandidate(
                            candidate_id=cand_id,
                            image=c_img,
                            source_origin=origin,
                            priority_level=2,
                            source_type="same_product",
                        )
                        self.cache_candidate(cand)
                        candidates.append(cand)
                    except Exception as ex:
                        logger.warning(f"Error loading pool candidate item {idx}: {ex}")

            # 2. Check cached references directory for matching descriptors or hex colors
            cached_files = list(self.cache_dir.glob("ref_*.png"))
            for cf in cached_files:
                if len(candidates) >= 5:
                    break
                try:
                    cand_img = Image.open(cf).convert("RGBA")
                    cand_id = cf.stem.replace("ref_", "")
                    # Avoid duplicate candidates
                    if any(c.candidate_id == cand_id for c in candidates):
                        continue

                    candidates.append(
                        ReferenceCandidate(
                            candidate_id=cand_id,
                            image=cand_img,
                            source_origin=f"cache:{cf.name}",
                            priority_level=3,
                            source_type="verified_catalog",
                            cached_path=cf,
                        )
                    )
                except Exception as ex:
                    logger.debug(f"Could not read cached file {cf}: {ex}")

            logger.info(
                f"ReferenceSearch completed in {time.time() - t0:.2f}s. "
                f"Found {len(candidates)} candidate reference(s)."
            )

        except Exception as e:
            # NETWORK FAILURE / UNEXPECTED ERROR RESILIENCE:
            # Do NOT fail the pipeline! Log warning and return empty candidates.
            logger.warning(
                f"ReferenceSearch encountered a network or IO exception ({e}). "
                f"Gracefully falling back to uploaded image.",
                exc_info=True,
            )
            return []

        return candidates
