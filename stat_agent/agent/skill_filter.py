"""Programmatic skill filtering based on data format/modality/level requirements."""
from __future__ import annotations

import logging
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from stat_agent.core.session import SimpleSession
    from stat_agent.agent.skill_registry import SkillMetadata

logger = logging.getLogger(__name__)


class SkillFilter:
    """Filters skills based on data format, modality, and data level requirements.

    This is Phase 1 filtering - programmatic constraints before LLM semantic matching.
    Takes planner output (which slices to analyze) and filters skills accordingly.

    Examples
    --------
    >>> filter = SkillFilter()
    >>> compatible = filter.filter_skills(
    ...     target_slice_ids=[0],
    ...     session=session,
    ...     all_skills=registry.skill_metadata
    ... )
    """

    def filter_skills(
        self,
        target_slice_ids: List[int],
        session: SimpleSession,
        all_skills: Dict[str, SkillMetadata]
    ) -> List[SkillMetadata]:
        """Filter skills compatible with target slices.

        Parameters
        ----------
        target_slice_ids : List[int]
            Slice IDs to analyze (from planner output)
        session : SimpleSession
            Current session with loaded data
        all_skills : Dict[str, SkillMetadata]
            All available skills from registry

        Returns
        -------
        List[SkillMetadata]
            Skills that are compatible with the target slices

        Notes
        -----
        Filtering logic per field:
        - ``num_slices``: Omit to allow any number of slices
        - ``modalities``: Omit to allow any modality. Single element broadcasts
          to all slices. Supports "/" as OR (e.g. "gene/protein").
        - ``data_levels``: Omit to allow any data level. Single element broadcasts
          to all slices. Supports "/" as OR (e.g. "cell/spot").
        """
        if not session or not session.has_data:
            logger.warning("No session data available for filtering")
            return list(all_skills.values())

        if not target_slice_ids:
            logger.warning("No target slices specified")
            return list(all_skills.values())

        # Get properties of target slices
        slice_properties = []
        for slice_id in target_slice_ids:
            slice_obj = session.get_slice(slice_id)
            if not slice_obj:
                logger.error(f"Slice {slice_id} not found in session")
                continue

            slice_properties.append({
                'slice_id': slice_id,
                'modality': slice_obj.modality,
                'data_level': slice_obj.data_level,
            })

        if len(slice_properties) != len(target_slice_ids):
            logger.error("Could not retrieve all target slice properties")
            return []

        # Filter skills
        compatible_skills = []
        filtered_out = []

        for slug, skill in all_skills.items():
            # Skip skills without filter requirements (backward compatibility)
            if not skill.filter_requirements:
                logger.debug(f"Skill '{slug}' has no filter_requirements, including by default")
                compatible_skills.append(skill)
                continue

            # Check if skill is compatible
            is_compatible, reason = self._check_compatibility(
                skill=skill,
                num_slices=len(target_slice_ids),
                slice_properties=slice_properties
            )

            if is_compatible:
                compatible_skills.append(skill)
            else:
                filtered_out.append((slug, reason))
                logger.debug(f"Filtered out '{slug}': {reason}")

        # Log filtering summary
        if filtered_out:
            logger.info(
                f"Filtered {len(filtered_out)}/{len(all_skills)} skills "
                f"for {len(target_slice_ids)} slice(s): {target_slice_ids}"
            )
            for slug, reason in filtered_out:
                logger.debug(f"  - {slug}: {reason}")

        logger.info(
            f"Filter result: {len(compatible_skills)} compatible skills for "
            f"slices {target_slice_ids}"
        )

        return compatible_skills

    def _check_compatibility(
        self,
        skill: SkillMetadata,
        num_slices: int,
        slice_properties: List[Dict]
    ) -> tuple[bool, str]:
        """Check if skill is compatible with target slices.

        Parameters
        ----------
        skill : SkillMetadata
            Skill to check
        num_slices : int
            Number of target slices
        slice_properties : List[Dict]
            Properties of each target slice

        Returns
        -------
        (is_compatible, reason) : tuple[bool, str]
            Whether skill is compatible and reason if not

        Notes
        -----
        Filtering rules for each field:
        - Omitted / empty: no constraint (skip check)
        - Single element list (e.g. [gene]): broadcast — ALL slices must match
        - List length == num_slices: positional — slice i must match element i
        - Values support "/" as OR (e.g. "cell/spot" matches either)
        """
        filter_req = skill.filter_requirements

        # Check 1: Number of slices (omit to allow any number)
        required_num_slices = filter_req.get('num_slices')
        if required_num_slices is not None:
            if num_slices != required_num_slices:
                return False, f"requires {required_num_slices} slice(s), got {num_slices}"

        # Check 2: Modalities (omit to allow any modality)
        required_modalities = filter_req.get('modalities', [])
        if required_modalities:
            # Single element: broadcast to all slices
            if len(required_modalities) == 1:
                required_modalities = required_modalities * num_slices
            elif len(required_modalities) != num_slices:
                return False, f"modality list length mismatch: requires {len(required_modalities)}, got {num_slices} slices"

            for i, (required_mod, slice_prop) in enumerate(zip(required_modalities, slice_properties)):
                actual_mod = slice_prop['modality']
                # Support "gene/protein" meaning "gene or protein"
                allowed = [m.strip() for m in required_mod.split('/')]
                if actual_mod not in allowed:
                    slice_id = slice_prop['slice_id']
                    return False, f"slice {slice_id} has modality '{actual_mod}', requires '{required_mod}'"

        # Check 3: Data levels (omit to allow any data level)
        required_levels = filter_req.get('data_levels', [])
        if required_levels:
            # Single element: broadcast to all slices
            if len(required_levels) == 1:
                required_levels = required_levels * num_slices
            elif len(required_levels) != num_slices:
                return False, f"data_levels list length mismatch: requires {len(required_levels)}, got {num_slices} slices"

            for i, (required_level, slice_prop) in enumerate(zip(required_levels, slice_properties)):
                actual_level = slice_prop['data_level']
                # Support "cell/spot" meaning "cell or spot"
                allowed = [l.strip() for l in required_level.split('/')]
                if actual_level not in allowed:
                    slice_id = slice_prop['slice_id']
                    return False, f"slice {slice_id} has data_level '{actual_level}', requires '{required_level}'"

        return True, ""

    def get_filter_summary(
        self,
        target_slice_ids: List[int],
        session: SimpleSession
    ) -> Dict:
        """Get summary of what filtering will check.

        Useful for debugging and logging.

        Parameters
        ----------
        target_slice_ids : List[int]
            Target slice IDs
        session : SimpleSession
            Current session

        Returns
        -------
        Dict
            Summary with num_slices, modalities, data_levels
        """
        summary = {
            'num_slices': len(target_slice_ids),
            'modalities': [],
            'data_levels': [],
            'slice_ids': target_slice_ids
        }

        for slice_id in target_slice_ids:
            slice_obj = session.get_slice(slice_id)
            if slice_obj:
                summary['modalities'].append(slice_obj.modality)
                summary['data_levels'].append(slice_obj.data_level)

        return summary
