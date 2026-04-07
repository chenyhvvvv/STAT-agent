"""Utilities for loading and routing Spatial Transcriptomics Agent Skills."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional YAML support for robust frontmatter parsing
try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    yaml = None
    _YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


class SkillInstructionFormatter:
    """Formats skill instructions optimally for different LLM providers.

    Each LLM family has different strengths in processing instructions:
    - GPT: Excels at structured, step-by-step instructions
    - Gemini: Prefers concise, logically-flowing descriptions
    - Claude: Understands natural language with implicit context
    - Others: Default to explicit, detailed instructions
    """

    PROVIDER_STYLES = {
        'openai': 'structured',
        'gpt': 'structured',
        'google': 'concise',
        'gemini': 'concise',
        'anthropic': 'natural',
        'claude': 'natural',
        'deepseek': 'explicit',
        'qwen': 'explicit',
        'default': 'explicit'
    }

    @classmethod
    def format_for_provider(cls, skill_body: str, provider: Optional[str] = None, max_chars: int = 4000) -> str:
        """Format skill instructions based on LLM provider.

        Args:
            skill_body: Raw skill instruction text
            provider: LLM provider name (openai, google, anthropic, etc.)
            max_chars: Maximum characters to return

        Returns:
            Formatted instruction text optimized for the provider
        """
        text = (skill_body or "").strip()
        if not text:
            return ""

        provider_key = (provider or 'default').lower()
        style = cls.PROVIDER_STYLES.get(provider_key, cls.PROVIDER_STYLES['default'])

        if style == 'structured':
            formatted = cls._add_structure_markers(text)
        elif style == 'concise':
            formatted = cls._make_concise(text)
        elif style == 'natural':
            formatted = text
        else:  # explicit
            formatted = cls._add_explicit_details(text)

        if len(formatted) <= max_chars:
            return formatted
        return formatted[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _add_structure_markers(text: str) -> str:
        """Add structured markers for GPT-style processing."""
        # Already numbered headings present
        if re.search(r'^\d+\.', text, re.MULTILINE):
            return text

        lines = text.split('\n')
        structured: List[str] = []
        for line in lines:
            if line.startswith('##'):
                structured.append(f"\n{line.upper()}\n")
            else:
                structured.append(line)
        return '\n'.join(structured)

    @staticmethod
    def _make_concise(text: str) -> str:
        """Simplify for Gemini-style processing."""
        if len(text) > 2000:
            parts = re.split(r'\n## Example', text)
            if len(parts) > 2:
                text = parts[0] + '\n## Example' + parts[1]
        return text

    @staticmethod
    def _add_explicit_details(text: str) -> str:
        """Add explicit details for general LLMs."""
        return re.sub(r'\n## (Usage|How to|Best Practices)', r'\n## IMPORTANT: \1', text)


@dataclass
class SkillMetadata:
    """Lightweight skill metadata for progressive disclosure (name + description only).

    This is loaded at startup to enable LLM-based skill matching without loading full content.
    """
    name: str
    slug: str
    description: str
    path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW: Filter requirements (programmatic filtering)
    filter_requirements: Optional[Dict[str, Any]] = None  # {num_slices, modalities, data_levels}

    # NEW: Prerequisites (simple descriptions for verifier)
    prerequisites: List[str] = field(default_factory=list)

    # Whether this skill is enabled by default
    default_skill: bool = True

    # Legacy format-aware fields (kept for backward compatibility)
    applicable_formats: List[str] = field(default_factory=lambda: ["single_slice", "multi_slice", "multi_omics"])
    modality_required: Optional[str] = None  # 'gene', 'protein', or None (either)
    slice_behavior: str = "per_slice"  # 'per_slice', 'cross_slice', 'any'

    def to_dict(self) -> Dict[str, str]:
        """Convert metadata to dictionary for LLM consumption."""
        return {
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
        }

    def is_compatible_with_format(self, data_format: str) -> bool:
        """Check if skill is compatible with given data format.

        Args:
            data_format: 'single_slice', 'multi_slice', or 'multi_omics'

        Returns:
            True if skill can work with this format
        """
        return data_format in self.applicable_formats

    def is_compatible_with_modality(self, current_modality: str) -> bool:
        """Check if skill is compatible with current modality.

        Args:
            current_modality: 'gene' or 'protein'

        Returns:
            True if skill can work with this modality (or if no modality requirement)
        """
        if self.modality_required is None:
            return True  # No modality requirement = works with any
        return current_modality == self.modality_required


@dataclass
class SkillDefinition:
    """Represents a single Agent Skill with full content loaded.

    name: display title; slug: lowercase-hyphen identifier for routing.
    """

    name: str
    slug: str
    description: str
    path: Path
    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW: Filter requirements (programmatic filtering)
    filter_requirements: Optional[Dict[str, Any]] = None  # {num_slices, modalities, data_levels}

    # NEW: Prerequisites (simple descriptions for verifier)
    prerequisites: List[str] = field(default_factory=list)

    # Whether this skill is enabled by default
    default_skill: bool = True

    # Legacy format-aware fields (kept for backward compatibility)
    applicable_formats: List[str] = field(default_factory=lambda: ["single_slice", "multi_slice", "multi_omics"])
    modality_required: Optional[str] = None  # 'gene', 'protein', or None (either)
    slice_behavior: str = "per_slice"  # 'per_slice', 'cross_slice', 'any'

    def prompt_instructions(self, max_chars: int = 4000, provider: Optional[str] = None) -> str:
        """Return the main instruction body, formatted for the LLM provider.

        Examples
        --------
        >>> skill = SkillDefinition(
        ...     name="QC",
        ...     slug="qc",
        ...     description="Quality control workflow",
        ...     path=Path("/tmp/qc"),
        ...     body="---\\ntitle: QC\\nslug: qc\\n---\\n## Steps\\n- filter\\n- score",
        ...     metadata={"title": "QC"},
        ... )
        >>> snippet = skill.prompt_instructions(max_chars=16)
        >>> isinstance(snippet, str)
        True
        """
        return SkillInstructionFormatter.format_for_provider(self.body, provider=provider, max_chars=max_chars)

    @property
    def summary_text(self) -> str:
        """Combine metadata and first section for lightweight scoring."""

        header = f"{self.name}\n{self.description}\n"
        primary_section = self.body.split("\n\n", 1)[0]
        return header + primary_section


@dataclass
class SkillMatch:
    """Represents a routing decision for a query."""

    skill: SkillDefinition
    score: float

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.skill.name,  # display title
            "slug": self.skill.slug,
            "score": f"{self.score:.3f}",
            "description": self.skill.description,
            "path": str(self.skill.path),
        }


class SkillRegistry:
    """Loads skills from the filesystem with progressive disclosure.

    At startup, only loads name + description (SkillMetadata) for fast LLM-based matching.
    Full skill content (SkillDefinition with body) is loaded on-demand when needed.
    """

    def __init__(self, skill_root: Path, progressive_disclosure: bool = True):
        self.skill_root = skill_root
        self.progressive_disclosure = progressive_disclosure
        self._skill_metadata: Dict[str, SkillMetadata] = {}
        self._full_skills_cache: Dict[str, SkillDefinition] = {}

    @property
    def skills(self) -> Dict[str, SkillDefinition]:
        """Backward compatibility: return full skills, loading them if needed."""
        if not self.progressive_disclosure:
            return self._full_skills_cache

        # If using progressive disclosure, load all skills on first access
        for slug in self._skill_metadata:
            if slug not in self._full_skills_cache:
                self.load_full_skill(slug)
        return self._full_skills_cache

    @property
    def skill_metadata(self) -> Dict[str, SkillMetadata]:
        """Get lightweight skill metadata (name + description only) for LLM matching."""
        return self._skill_metadata

    def load(self) -> None:
        """Discover skills under the configured skill root.

        With progressive_disclosure=True (default): Only loads name + description.
        With progressive_disclosure=False: Loads full skill content.
        """

        if not self.skill_root.exists():
            logger.warning("Skill root %s does not exist; no skills loaded.", self.skill_root)
            self._skill_metadata = {}
            self._full_skills_cache = {}
            return

        if self.progressive_disclosure:
            # Load only metadata (name + description) for fast startup
            discovered_metadata: Dict[str, SkillMetadata] = {}
            for skill_file in sorted(self.skill_root.glob("*/SKILL.md")):
                metadata = self._parse_skill_metadata(skill_file)
                if not metadata:
                    continue
                key = metadata.slug.lower()
                if key in discovered_metadata:
                    logger.warning("Duplicate skill name '%s' found; keeping first occurrence.", metadata.name)
                    continue
                discovered_metadata[key] = metadata
                logger.info("Loaded skill metadata '%s' from %s", metadata.name, skill_file)
            self._skill_metadata = discovered_metadata
        else:
            # Load full skill content (backward compatibility mode)
            discovered: Dict[str, SkillDefinition] = {}
            for skill_file in sorted(self.skill_root.glob("*/SKILL.md")):
                definition = self._parse_skill_file(skill_file)
                if not definition:
                    continue
                key = definition.slug.lower()
                if key in discovered:
                    logger.warning("Duplicate skill name '%s' found; keeping first occurrence.", definition.name)
                    continue
                discovered[key] = definition
                logger.info("Loaded skill '%s' from %s", definition.name, skill_file)
            self._full_skills_cache = discovered
            # Also populate metadata
            self._skill_metadata = {
                slug: SkillMetadata(
                    name=skill.name,
                    slug=skill.slug,
                    description=skill.description,
                    path=skill.path,
                    metadata=skill.metadata,
                    filter_requirements=skill.filter_requirements,
                    prerequisites=skill.prerequisites,
                    default_skill=skill.default_skill,
                    applicable_formats=skill.applicable_formats,
                    modality_required=skill.modality_required,
                    slice_behavior=skill.slice_behavior
                )
                for slug, skill in discovered.items()
            }

    def load_full_skill(self, slug: str) -> Optional[SkillDefinition]:
        """Lazy-load full skill content (body) for a specific skill.

        This is called on-demand when the LLM decides to use a skill.

        Args:
            slug: Skill slug identifier

        Returns:
            SkillDefinition with full body, or None if not found
        """
        slug_lower = slug.lower()

        # Return from cache if already loaded
        if slug_lower in self._full_skills_cache:
            return self._full_skills_cache[slug_lower]

        # Get metadata to find the skill file
        metadata = self._skill_metadata.get(slug_lower)
        if not metadata:
            logger.warning("Skill '%s' not found in registry", slug)
            return None

        # Parse the full skill file
        skill_file = metadata.path / "SKILL.md"
        definition = self._parse_skill_file(skill_file)
        if definition:
            self._full_skills_cache[slug_lower] = definition
            logger.info("Loaded full skill content for '%s'", metadata.name)
        return definition

    def _parse_skill_metadata(self, skill_file: Path) -> Optional[SkillMetadata]:
        """Parse only the frontmatter (name + description) without loading full body.

        This enables fast startup with progressive disclosure.
        """
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Unable to read skill file %s: %s", skill_file, exc)
            return None

        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            logger.warning("Skill file %s is missing YAML frontmatter.", skill_file)
            return None

        try:
            closing_index = lines.index("---", 1)
        except ValueError:
            logger.warning("Skill file %s has unterminated YAML frontmatter.", skill_file)
            return None

        frontmatter_lines = lines[1:closing_index]
        metadata = self._parse_frontmatter(frontmatter_lines)
        raw_name = metadata.get("name")
        description = metadata.get("description")
        # Determine display title and slug with backward compatibility
        title = metadata.get("title") or metadata.get("display_title") or raw_name
        slug_value = metadata.get("slug")
        if not slug_value:
            # If name is slug-like, use it; otherwise slugify the title
            slug_value = raw_name if self._looks_like_slug(raw_name) else self._slugify(title)
        if not (title and description and slug_value):
            logger.warning("Skill file %s is missing required title/description/slug metadata.", skill_file)
            return None

        skill_path = skill_file.parent

        # Parse format-aware fields (Phase 5: multi-format support)
        applicable_formats = self._parse_list_field(metadata, "applicable_formats", default=["single_slice", "multi_slice", "multi_omics"])
        modality_required = metadata.get("modality_required")  # 'gene', 'protein', or None
        slice_behavior = metadata.get("slice_behavior", "per_slice")

        # NEW: Parse filter requirements
        filter_requirements = self._parse_filter_requirements(metadata)

        # NEW: Parse prerequisites (list of simple descriptions)
        prerequisites = self._parse_list_field(metadata, "prerequisites", default=[])

        # Parse default_skill (default True if missing)
        default_skill = metadata.get("default_skill", True)
        if isinstance(default_skill, str):
            default_skill = default_skill.lower() in ('true', '1', 'yes')

        return SkillMetadata(
            name=str(title),
            slug=str(slug_value),
            description=str(description),
            path=skill_path,
            metadata=metadata,
            filter_requirements=filter_requirements,
            prerequisites=prerequisites,
            default_skill=bool(default_skill),
            applicable_formats=applicable_formats,
            modality_required=modality_required,
            slice_behavior=slice_behavior
        )

    def _parse_skill_file(self, skill_file: Path) -> Optional[SkillDefinition]:
        try:
            content = skill_file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Unable to read skill file %s: %s", skill_file, exc)
            return None

        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            logger.warning("Skill file %s is missing YAML frontmatter.", skill_file)
            return None

        try:
            closing_index = lines.index("---", 1)
        except ValueError:
            logger.warning("Skill file %s has unterminated YAML frontmatter.", skill_file)
            return None

        frontmatter_lines = lines[1:closing_index]
        metadata = self._parse_frontmatter(frontmatter_lines)
        raw_name = metadata.get("name")
        description = metadata.get("description")
        # Determine display title and slug with backward compatibility
        title = metadata.get("title") or metadata.get("display_title") or raw_name
        slug_value = metadata.get("slug")
        if not slug_value:
            # If name is slug-like, use it; otherwise slugify the title
            slug_value = raw_name if self._looks_like_slug(raw_name) else self._slugify(title)
        if not (title and description and slug_value):
            logger.warning("Skill file %s is missing required title/description/slug metadata.", skill_file)
            return None

        body = "\n".join(lines[closing_index + 1 :]).strip()
        skill_path = skill_file.parent

        # Parse format-aware fields (Phase 5: multi-format support)
        applicable_formats = self._parse_list_field(metadata, "applicable_formats", default=["single_slice", "multi_slice", "multi_omics"])
        modality_required = metadata.get("modality_required")
        slice_behavior = metadata.get("slice_behavior", "per_slice")

        # NEW: Parse filter requirements
        filter_requirements = self._parse_filter_requirements(metadata)

        # NEW: Parse prerequisites (list of simple descriptions)
        prerequisites = self._parse_list_field(metadata, "prerequisites", default=[])

        # Parse default_skill (default True if missing)
        default_skill = metadata.get("default_skill", True)
        if isinstance(default_skill, str):
            default_skill = default_skill.lower() in ('true', '1', 'yes')

        return SkillDefinition(
            name=str(title),
            slug=str(slug_value),
            description=str(description),
            path=skill_path,
            body=body,
            metadata=metadata,
            filter_requirements=filter_requirements,
            prerequisites=prerequisites,
            default_skill=bool(default_skill),
            applicable_formats=applicable_formats,
            modality_required=modality_required,
            slice_behavior=slice_behavior
        )

    @staticmethod
    def _parse_list_field(metadata: Dict[str, Any], field_name: str, default: List[str]) -> List[str]:
        """Parse a list field from metadata, handling both list and string values.

        Args:
            metadata: Parsed frontmatter dictionary
            field_name: Name of the field to parse
            default: Default value if field is missing

        Returns:
            List of string values
        """
        value = metadata.get(field_name)
        if value is None:
            return default

        # If already a list, return it
        if isinstance(value, list):
            return [str(item) for item in value]

        # If string, try to split by newlines or commas
        if isinstance(value, str):
            # Handle multi-line strings (from YAML multiline)
            if '\n' in value:
                return [line.strip() for line in value.split('\n') if line.strip() and not line.strip().startswith('#')]
            # Handle comma-separated
            if ',' in value:
                return [item.strip() for item in value.split(',') if item.strip()]
            # Single value
            return [value.strip()]

        return default

    @staticmethod
    def _parse_frontmatter(lines: Iterable[str]) -> Dict[str, Any]:
        """Parse YAML frontmatter preserving types.

        Prefers yaml.safe_load if PyYAML is available to support multiline
        values and rich YAML constructs. Falls back to a minimal line-based
        parser if PyYAML is not installed.
        """

        # Try robust YAML parsing first
        if _YAML_AVAILABLE:
            text = "\n".join(list(lines))
            try:
                loaded: Optional[Dict[str, Any]] = yaml.safe_load(text)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to parse YAML frontmatter with PyYAML: %s", exc)
                loaded = None

            if isinstance(loaded, dict):
                # Return as-is to preserve types (dicts, lists, etc.)
                return loaded

        # Fallback: simple line-based parser (single-line key: value pairs only)
        metadata: Dict[str, str] = {}
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"')
        if not _YAML_AVAILABLE:
            logger.debug("PyYAML not installed; used fallback frontmatter parser.")
        return metadata

    @staticmethod
    def _parse_filter_requirements(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse filter_requirements field from metadata.

        Expected format:
        filter_requirements:
          num_slices: 1
          modalities: [gene]
          data_levels: [cell]

        Returns:
            Dictionary with filter requirements, or None if not present
        """
        filter_req = metadata.get("filter_requirements")
        if not filter_req:
            return None

        # If it's already a dict (from YAML), return it
        if isinstance(filter_req, dict):
            return filter_req

        # If it's a string (from fallback parser), try to parse it
        # This shouldn't happen with PyYAML, but handle gracefully
        logger.warning("filter_requirements should be a dictionary, got: %s", type(filter_req))
        return None

    @staticmethod
    def _looks_like_slug(value: Optional[str]) -> bool:
        if not value or not isinstance(value, str):
            return False
        return re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is not None

    @staticmethod
    def _slugify(value: Optional[str], max_len: int = 64) -> str:
        if not value:
            return ""
        slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
        slug = re.sub(r"-+", "-", slug)
        if len(slug) > max_len:
            slug = slug[:max_len].strip("-")
        return slug


class SkillRouter:
    """Simple keyword-based router that ranks skills for a query."""

    def __init__(self, registry: SkillRegistry, min_score: float = 0.1):
        self.registry = registry
        self.min_score = min_score
        self._skill_vectors: Dict[str, Dict[str, int]] = {}
        self._build_vectors()

    def _build_vectors(self) -> None:
        self._skill_vectors = {
            key: self._token_frequency(definition.summary_text)
            for key, definition in self.registry.skills.items()
        }

    def refresh(self) -> None:
        self._build_vectors()

    def route(self, query: str, top_k: int = 1) -> List[SkillMatch]:
        if not query or not query.strip():
            return []
        query_vector = self._token_frequency(query)
        if not query_vector:
            return []

        scored: List[Tuple[str, float]] = []
        for key, skill_vector in self._skill_vectors.items():
            score = self._cosine_similarity(query_vector, skill_vector)
            scored.append((key, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        matches: List[SkillMatch] = []
        for key, score in scored[:top_k]:
            if score < self.min_score:
                continue
            skill = self.registry.skills.get(key)
            if not skill:
                continue
            matches.append(SkillMatch(skill=skill, score=score))
        return matches

    @staticmethod
    def _token_frequency(text: str) -> Dict[str, int]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        freq: Dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        return freq

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        common = set(vec_a.keys()) & set(vec_b.keys())
        numerator = sum(vec_a[token] * vec_b[token] for token in common)
        if numerator == 0:
            return 0.0
        sum_sq_a = sum(value * value for value in vec_a.values())
        sum_sq_b = sum(value * value for value in vec_b.values())
        denominator = (sum_sq_a ** 0.5) * (sum_sq_b ** 0.5)
        if denominator == 0:
            return 0.0
        return numerator / denominator


def build_skill_registry(project_root: Path) -> SkillRegistry:
    """Helper to create and load a registry from the project root."""

    skill_root = project_root / ".claude" / "skills"
    registry = SkillRegistry(skill_root=skill_root)
    registry.load()
    if not registry.skills:
        logger.warning("No skills discovered under %s", skill_root)
    return registry


def build_multi_path_skill_registry(package_root: Path, cwd: Path) -> SkillRegistry:
    """
    Load skills from multiple paths with priority ordering.

    Searches for skills in:
    1. Package root (.claude/skills in package installation directory) - built-in skills
    2. Current working directory (.claude/skills in user's project) - user-created skills

    User-created skills (CWD) take priority over built-in skills (package) if there are duplicates.

    Always returns a SkillRegistry instance, even when no skills are discovered.
    """
    package_skill_root = package_root / ".claude" / "skills"
    cwd_skill_root = cwd / ".claude" / "skills"

    pkg = SkillRegistry(skill_root=package_skill_root)
    pkg.load()
    usr = SkillRegistry(skill_root=cwd_skill_root)
    usr.load()

    merged: Dict[str, SkillDefinition] = {}
    if pkg.skills:
        merged.update(pkg.skills)
        logger.info("Loaded %d built-in skills from %s", len(pkg.skills), package_skill_root)
    if usr.skills:
        for slug, defn in usr.skills.items():
            if slug in merged:
                logger.info("User skill '%s' overrides built-in skill", defn.name)
            merged[slug] = defn
        logger.info("Loaded %d user skills from %s", len(usr.skills), cwd_skill_root)

    if not merged:
        logger.warning("No skills discovered in package or CWD")
        reg = SkillRegistry(skill_root=package_skill_root)
        reg._full_skills_cache = {}
        reg._skill_metadata = {}
        return reg

    reg = SkillRegistry(skill_root=package_skill_root)
    reg._full_skills_cache = merged
    # Also populate _skill_metadata so that skill_metadata property works correctly
    reg._skill_metadata = {
        slug: SkillMetadata(
            name=skill.name,
            slug=skill.slug,
            description=skill.description,
            path=skill.path,
            metadata=skill.metadata,
            filter_requirements=skill.filter_requirements,
            prerequisites=skill.prerequisites,
            applicable_formats=skill.applicable_formats,
            modality_required=skill.modality_required,
            slice_behavior=skill.slice_behavior
        )
        for slug, skill in merged.items()
    }
    return reg

__all__ = [
    "SkillInstructionFormatter",
    "SkillMetadata",
    "SkillDefinition",
    "SkillMatch",
    "SkillRegistry",
    "SkillRouter",
    "build_skill_registry",
    "build_multi_path_skill_registry",
]
