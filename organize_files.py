import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CategoryConfig:
    """Configuration for file categorization."""

    # Directory structure mapping
    DIRECTORY_STRUCTURE = {
        "audio_transcripts": {
            "base": "01_Audio_Transcripts",
            "subdirs": {
                "podcast_episodes": "Podcast_Episodes",
                "radio_kmud": "Radio_Shows_KMUD",
                "politics_science": "Politics_Science",
                "other_interviews": "Other_Interviews"
            }
        },
        "publications": {
            "base": "02_Publications",
            "subdirs": {
                "townsend_letters": "Townsend_Letters",
                "academic_papers": "Academic_Papers"
            }
        },
        "chronological": {
            "base": "03_Chronological_Content",
            "subdirs": {
                "1980s": "1980s",
                "1990s": "1990s",
                "2000s": "2000s",
                "2010s": "2010s"
            }
        },
        "health_topics": {
            "base": "04_Health_Topics",
            "subdirs": {
                "hormones": "Hormones_Endocrine",
                "metabolism": "Metabolism_Energy",
                "aging": "Aging_Degenerative",
                "cancer": "Cancer_Research",
                "nutrition": "Nutrition_Diet"
            }
        },
        "academic_docs": "05_Academic_Documents",
        "email_comms": "06_Email_Communications",
        "special_collections": "07_Special_Collections",
        "newsletters": "08_Newslatters"
    }

    # Keyword mappings for content-based categorization
    HEALTH_TOPIC_KEYWORDS = {
        "hormones": [
            "estrogen", "progesterone", "thyroid", "hormone", "dhea",
            "testosterone", "cortisol", "insulin", "melatonin"
        ],
        "metabolism": [
            "metabolism", "energy", "glucose", "sugar", "mitochondria",
            "co2", "respiration", "atp", "glycolysis"
        ],
        "aging": [
            "aging", "degeneration", "inflammation", "stress", "fatigue",
            "longevity", "senescence"
        ],
        "nutrition": [
            "nutrition", "diet", "food", "milk", "fats", "oils", "vitamin",
            "mineral", "supplement", "protein", "carbohydrate"
        ]
    }

    # Special pattern keywords
    SPECIAL_KEYWORDS = {
        "academic": ["thesis", "phd", "dissertation", "research"],
        "email": ["email", "exchanges", "correspondence"],
        "special_collections": ["lost conversations", "interviews revisited", "collection"]
    }


class FileOrganizer:
    """Main file organization class with improved categorization logic."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.config = CategoryConfig()
        self.move_summary: Dict[str, List[str]] = {}

    def create_directory_structure(self) -> None:
        """Create the complete directory structure."""
        directories_to_create = []

        # Build flat list of all directories
        for category, structure in self.config.DIRECTORY_STRUCTURE.items():
            if isinstance(structure, dict):
                # Handle nested structure
                base_dir = structure["base"]
                directories_to_create.append(base_dir)

                if "subdirs" in structure:
                    for subdir in structure["subdirs"].values():
                        directories_to_create.append(f"{base_dir}/{subdir}")
            else:
                # Handle simple string structure
                directories_to_create.append(structure)

        # Create all directories
        for directory in directories_to_create:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {dir_path}")

    def _extract_year_from_filename(self, filename: str) -> Optional[int]:
        """Extract year from chronological filename pattern."""
        year_pattern = r'^(\d{4})\s*-\s*\w+\.txt$'
        match = re.match(year_pattern, filename)
        return int(match.group(1)) if match else None

    def _categorize_by_keywords(self, filename_lower: str, keyword_dict: Dict[str, List[str]]) -> Optional[str]:
        """Categorize file based on keyword matching."""
        for category, keywords in keyword_dict.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        return None

    def categorize_file(self, filename: str) -> str:
        """
        Determine the appropriate category for a file using improved logic.

        Args:
            filename: Name of the file to categorize

        Returns:
            Relative path to the target directory
        """
        filename_lower = filename.lower()

        # Priority 1: Audio Transcripts (highest specificity)
        if filename.endswith('.mp3-transcript.txt'):
            return self._categorize_audio_transcript(filename_lower)

        # Priority 2: Special document types
        special_category = self._categorize_special_documents(filename_lower)
        if special_category:
            return special_category

        # Priority 3: Chronological content
        year = self._extract_year_from_filename(filename)
        if year:
            return self._categorize_by_year(year)

        # Priority 4: Health topics (content-based)
        if filename.endswith(('.html', '.txt')):
            health_category = self._categorize_health_topics(filename_lower)
            if health_category:
                return health_category

            # Cancer gets special treatment
            if 'cancer' in filename_lower:
                return f"{self.config.DIRECTORY_STRUCTURE['health_topics']['base']}/{self.config.DIRECTORY_STRUCTURE['health_topics']['subdirs']['cancer']}"

            # General academic papers
            if filename.endswith('.html') or any(term in filename_lower for term in ['physiology', 'biological', 'medical', 'research']):
                return f"{self.config.DIRECTORY_STRUCTURE['publications']['base']}/{self.config.DIRECTORY_STRUCTURE['publications']['subdirs']['academic_papers']}"

        # Default: newsletters
        return self.config.DIRECTORY_STRUCTURE['newsletters']

    def _categorize_audio_transcript(self, filename_lower: str) -> str:
        """Categorize audio transcript files."""
        audio_base = self.config.DIRECTORY_STRUCTURE['audio_transcripts']['base']
        subdirs = self.config.DIRECTORY_STRUCTURE['audio_transcripts']['subdirs']

        if 'kmud-' in filename_lower:
            return f"{audio_base}/{subdirs['radio_kmud']}"
        elif 'polsci-' in filename_lower:
            return f"{audio_base}/{subdirs['politics_science']}"
        elif filename_lower.startswith('#') and '∩' in filename_lower:
            return f"{audio_base}/{subdirs['podcast_episodes']}"
        else:
            return f"{audio_base}/{subdirs['other_interviews']}"

    def _categorize_special_documents(self, filename_lower: str) -> Optional[str]:
        """Categorize special document types."""
        if 'townsend letter for doctors' in filename_lower:
            return f"{self.config.DIRECTORY_STRUCTURE['publications']['base']}/{self.config.DIRECTORY_STRUCTURE['publications']['subdirs']['townsend_letters']}"

        if any(term in filename_lower for term in self.config.SPECIAL_KEYWORDS['academic']):
            return self.config.DIRECTORY_STRUCTURE['academic_docs']

        if any(term in filename_lower for term in self.config.SPECIAL_KEYWORDS['email']):
            return self.config.DIRECTORY_STRUCTURE['email_comms']

        if any(term in filename_lower for term in self.config.SPECIAL_KEYWORDS['special_collections']):
            return self.config.DIRECTORY_STRUCTURE['special_collections']

        return None

    def _categorize_by_year(self, year: int) -> str:
        """Categorize files by year."""
        base = self.config.DIRECTORY_STRUCTURE['chronological']['base']
        subdirs = self.config.DIRECTORY_STRUCTURE['chronological']['subdirs']

        decade_map = {
            range(1980, 1990): subdirs['1980s'],
            range(1990, 2000): subdirs['1990s'],
            range(2000, 2010): subdirs['2000s'],
            range(2010, 2020): subdirs['2010s']
        }

        for decade_range, subdir in decade_map.items():
            if year in decade_range:
                return f"{base}/{subdir}"

        return base  # Fallback for years outside defined ranges

    def _categorize_health_topics(self, filename_lower: str) -> Optional[str]:
        """Categorize health topic files."""
        base = self.config.DIRECTORY_STRUCTURE['health_topics']['base']
        subdirs = self.config.DIRECTORY_STRUCTURE['health_topics']['subdirs']

        topic_category = self._categorize_by_keywords(
            filename_lower, self.config.HEALTH_TOPIC_KEYWORDS)
        if topic_category:
            return f"{base}/{subdirs[topic_category]}"

        return None

    def _handle_file_conflicts(self, destination_path: Path) -> Path:
        """Handle file name conflicts by appending a counter."""
        if not destination_path.exists():
            return destination_path

        base_name = destination_path.stem
        extension = destination_path.suffix
        counter = 1

        while destination_path.exists():
            new_name = f"{base_name}_duplicate_{counter}{extension}"
            destination_path = destination_path.parent / new_name
            counter += 1

        return destination_path

    def _track_move(self, category: str, filename: str) -> None:
        """Track file moves for summary reporting."""
        if category not in self.move_summary:
            self.move_summary[category] = []
        self.move_summary[category].append(filename)

    def _move_file(self, source_path: Path, destination_path: Path, filename: str, category: str) -> bool:
        """Move a single file and handle errors."""
        try:
            final_destination = self._handle_file_conflicts(destination_path)
            shutil.move(str(source_path), str(final_destination))
            print(f"✅ MOVED: {filename} → {category}")
            return True
        except Exception as e:
            print(f"❌ ERROR moving {filename}: {e}")
            return False

    def organize_files(self, dry_run: bool = False) -> None:
        """
        Main file organization method.

        Args:
            dry_run: If True, only show what would be done without actually moving files
        """
        if not self.base_path.exists():
            print(f"❌ Error: Data directory {self.base_path} does not exist!")
            return

        # Create directory structure
        if not dry_run:
            self.create_directory_structure()

        # Get all files to process
        files = [f for f in self.base_path.iterdir()
                 if f.is_file() and not f.name.startswith('.') and f.name != '.DS_Store']

        print(f"\n📊 Processing {len(files)} files...")
        print("=" * 70)

        success_count = 0
        error_count = 0

        for file_path in files:
            filename = file_path.name
            category = self.categorize_file(filename)
            destination_dir = self.base_path / category
            destination_path = destination_dir / filename

            self._track_move(category, filename)

            if dry_run:
                print(f"📋 WOULD MOVE: {filename} → {category}")
                success_count += 1
            else:
                if self._move_file(file_path, destination_path, filename, category):
                    success_count += 1
                else:
                    error_count += 1

        self._print_summary(success_count, error_count, dry_run)

    def _print_summary(self, success_count: int, error_count: int, dry_run: bool) -> None:
        """Print organization summary."""
        mode = "DRY RUN" if dry_run else "EXECUTION"
        print(f"\n{'=' * 70}")
        print(f"📈 ORGANIZATION SUMMARY ({mode})")
        print(f"{'=' * 70}")
        print(f"✅ Successful: {success_count}")
        if error_count > 0:
            print(f"❌ Errors: {error_count}")

        print("\n📁 Files by Category:")
        for category, files in sorted(self.move_summary.items()):
            print(f"\n{category}: {len(files)} files")
            for file in sorted(files)[:3]:  # Show first 3 files
                print(f"  • {file}")
            if len(files) > 3:
                print(f"  ... and {len(files) - 3} more files")


def main() -> None:
    """Main function with interactive interface."""
    script_dir = Path(__file__).parent
    data_path = script_dir / "raw_data"

    print("🧬 Ray Peat Data File Organization Script")
    print("=" * 50)
    print(f"📂 Data directory: {data_path}")

    if not data_path.exists():
        print(f"❌ Error: Data directory not found at {data_path}")
        return

    organizer = FileOrganizer(data_path)

    # Interactive mode
    while True:
        choice = input("\n🔍 Perform dry run first? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print("\n--- 🔍 DRY RUN MODE ---")
            organizer.organize_files(dry_run=True)

            confirm = input(
                "\n▶️ Proceed with actual file moves? (y/n): ").lower().strip()
            if confirm in ['y', 'yes']:
                print("\n--- 📁 MOVING FILES ---")
                organizer.organize_files(dry_run=False)
                print("\n🎉 File organization completed!")
            else:
                print("⏸️ Operation cancelled.")
            break
        elif choice in ['n', 'no']:
            print("\n--- 📁 MOVING FILES ---")
            organizer.organize_files(dry_run=False)
            print("\n🎉 File organization completed!")
            break
        else:
            print("❓ Please enter 'y' or 'n'")


if __name__ == "__main__":
    main()
