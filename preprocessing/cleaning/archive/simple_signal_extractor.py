#!/usr/bin/env python3
"""
Simplified Ray Peat Signal Extraction

This creates a simpler approach that uses rule-based enhancement 
combined with basic pattern matching to extract Ray Peat signal.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

class SimpleRayPeatExtractor:
    """Rules-based Ray Peat signal extractor."""
    
    def __init__(self):
        # Ray Peat signature phrases and topics
        self.ray_peat_indicators = [
            "ray peat", "dr. peat", "dr peat", "peat says", "peat explained",
            "thyroid", "progesterone", "pufa", "polyunsaturated", 
            "bioenergetic", "mitochondria", "metabolism", "cortisol",
            "estrogen", "serotonin", "temperature", "glycogen",
            "saturated fat", "coconut oil", "orange juice", "aspirin"
        ]
        
        # Noise patterns to remove
        self.noise_patterns = [
            r'This free program is paid for by.*?\.',
            r'If you\'re not already a member.*?\.',
            r'You\'re listening to.*?\.',
            r'Welcome.*?to the show.*?\.',
            r'.*?radio.*?network.*?\.',
            r'.*?frequency transmission.*?\.',
            r'.*?uplifting soulful.*?\.',
            r'Need one more reason why.*?Safeway.*?\.',
            r'For every \$10 you spend.*?\.',
            r'.*?Cuisinart.*?\.',
            r'.*?stamp saver.*?\.',
            r'.*?website.*?\.com.*?\.',
            r'.*?support.*?website.*?\.',
            r'.*?check.*?website.*?\.',
            r'Previously.*?asked.*?\.',
            r'.*?phone.*?number.*?\d+-\d+-\d+.*?\.',
            r'.*?email.*?@.*?\.',
            r'What up.*?everyone.*?\.',
            r'This is.*?from.*?\.',
        ]
        
        # Speaker indicators
        self.ray_peat_speaker_patterns = [
            r'Ray[:\s]',
            r'Dr\.?\s*Peat[:\s]',
            r'Peat[:\s]',
            r'Ray Peat[:\s]',
        ]
        
        self.host_speaker_patterns = [
            r'Host[:\s]',
            r'Josh[:\s]',
            r'Patrick[:\s]',
            r'Jeannie[:\s]',
            r'Adam[:\s]',
        ]
    
    def extract_signal(self, content: str, source_file: str = "") -> Dict:
        """Extract Ray Peat signal using rule-based methods."""
        if not content or len(content.strip()) < 100:
            return self._empty_result("Content too short")
        
        # Clean noise
        cleaned_content = self._remove_noise(content)
        
        # Extract Ray Peat segments
        ray_peat_segments = self._extract_ray_peat_segments(cleaned_content)
        
        # Calculate signal quality
        signal_ratio = self._calculate_signal_ratio(content, cleaned_content)
        ray_peat_percentage = self._calculate_ray_peat_percentage(cleaned_content, ray_peat_segments)
        
        # Determine content type
        content_type = self._determine_content_type(source_file, content)
        
        # Extract topics
        topics = self._extract_topics(cleaned_content)
        
        result = {
            "extracted_content": "\n\n".join(ray_peat_segments) if ray_peat_segments else cleaned_content,
            "content_type": content_type,
            "source_analysis": {
                "original_length": len(content),
                "extracted_length": len("\n\n".join(ray_peat_segments)) if ray_peat_segments else len(cleaned_content),
                "signal_ratio": signal_ratio,
                "ray_peat_percentage": ray_peat_percentage
            },
            "bioenergetic_content": {
                "primary_topics": topics,
                "mechanisms_explained": self._extract_mechanisms(cleaned_content),
                "recommendations": self._extract_recommendations(cleaned_content),
                "research_mentioned": []
            },
            "quality_assessment": {
                "signal_density": self._assess_signal_density(signal_ratio, ray_peat_percentage),
                "educational_value": self._assess_educational_value(topics),
                "uniqueness": "contains_ray_peat_insights" if ray_peat_percentage > 20 else "basic_information",
                "completeness": "comprehensive" if len(cleaned_content) > 2000 else "partial"
            },
            "extraction_notes": f"Rule-based extraction from {content_type}"
        }
        
        return result
    
    def _remove_noise(self, content: str) -> str:
        """Remove commercial and noise content."""
        cleaned = content
        
        # Remove noise patterns
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove very short lines (likely noise)
        lines = cleaned.split('\n')
        substantial_lines = [line for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(substantial_lines)
    
    def _extract_ray_peat_segments(self, content: str) -> List[str]:
        """Extract segments where Ray Peat is speaking or being discussed."""
        segments = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            # Check if paragraph contains Ray Peat indicators
            ray_peat_score = 0
            for indicator in self.ray_peat_indicators:
                if indicator.lower() in paragraph.lower():
                    ray_peat_score += 1
            
            # If paragraph has Ray Peat content, include it
            if ray_peat_score > 0 and len(paragraph) > 100:
                # Add speaker attribution if missing
                attributed = self._add_speaker_attribution(paragraph)
                segments.append(attributed)
        
        return segments
    
    def _add_speaker_attribution(self, text: str) -> str:
        """Add clear speaker attribution to text."""
        # Check if already has attribution
        if any(re.search(pattern, text) for pattern in self.ray_peat_speaker_patterns + self.host_speaker_patterns):
            return text
        
        # Check if this sounds like Ray Peat speaking
        if any(phrase in text.lower() for phrase in ["i think", "i've found", "in my experience", "i recommend"]):
            # Look for context clues
            if any(topic in text.lower() for topic in ["thyroid", "progesterone", "pufa", "mitochondria"]):
                return f"**RAY PEAT:** {text}"
        
        # Default - assume it's about Ray Peat's ideas
        return text
    
    def _calculate_signal_ratio(self, original: str, cleaned: str) -> int:
        """Calculate signal-to-noise ratio."""
        if not original:
            return 0
        return min(100, int((len(cleaned) / len(original)) * 100))
    
    def _calculate_ray_peat_percentage(self, content: str, ray_peat_segments: List[str]) -> int:
        """Calculate percentage of content that's Ray Peat related."""
        if not content:
            return 0
        
        ray_peat_length = sum(len(segment) for segment in ray_peat_segments)
        total_length = len(content)
        
        return min(100, int((ray_peat_length / total_length) * 100))
    
    def _determine_content_type(self, source_file: str, content: str) -> str:
        """Determine the type of content."""
        filename = source_file.lower()
        
        if "transcript" in filename or "mp3" in filename:
            return "conversation"
        elif "cleaned.txt" in filename:
            if any(word in content.lower()[:500] for word in ["abstract", "introduction", "methodology"]):
                return "paper"
            else:
                return "article"
        else:
            return "article"
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract bioenergetic topics from content."""
        topics = []
        content_lower = content.lower()
        
        topic_keywords = {
            "thyroid metabolism": ["thyroid", "t3", "t4", "tsh", "hypothyroid"],
            "hormone balance": ["progesterone", "estrogen", "cortisol", "testosterone"],
            "fat metabolism": ["pufa", "polyunsaturated", "saturated fat", "coconut oil"],
            "cellular energy": ["mitochondria", "atp", "metabolism", "glycogen"],
            "stress physiology": ["cortisol", "adrenaline", "stress", "adaptation"],
            "nutrition": ["orange juice", "milk", "sugar", "protein"],
            "inflammation": ["serotonin", "histamine", "endotoxin", "aspirin"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics[:5]  # Limit to top 5 topics
    
    def _extract_mechanisms(self, content: str) -> List[str]:
        """Extract physiological mechanisms mentioned."""
        mechanisms = []
        
        # Look for mechanism descriptions
        mechanism_patterns = [
            r'(\w+\s+increases?\s+\w+)',
            r'(\w+\s+decreases?\s+\w+)', 
            r'(\w+\s+inhibits?\s+\w+)',
            r'(\w+\s+stimulates?\s+\w+)',
            r'(\w+\s+causes?\s+\w+)',
        ]
        
        for pattern in mechanism_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            mechanisms.extend(matches[:3])  # Limit matches per pattern
        
        return mechanisms[:10]  # Limit total mechanisms
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract practical recommendations."""
        recommendations = []
        
        # Look for recommendation patterns
        rec_patterns = [
            r'(should\s+\w+\s+\w+[^.]*)',
            r'(recommend\s+\w+[^.]*)',
            r'(avoid\s+\w+[^.]*)',
            r'(use\s+\w+\s+for[^.]*)',
            r'(eat\s+\w+[^.]*)',
            r'(take\s+\w+[^.]*)',
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            recommendations.extend(matches[:2])
        
        return recommendations[:8]
    
    def _assess_signal_density(self, signal_ratio: int, ray_peat_percentage: int) -> str:
        """Assess the density of Ray Peat signal."""
        if ray_peat_percentage >= 60 and signal_ratio >= 70:
            return "very_high"
        elif ray_peat_percentage >= 40 and signal_ratio >= 50:
            return "high"
        elif ray_peat_percentage >= 20 and signal_ratio >= 30:
            return "medium"
        else:
            return "low"
    
    def _assess_educational_value(self, topics: List[str]) -> str:
        """Assess educational value based on topic coverage."""
        if len(topics) >= 4:
            return "excellent"
        elif len(topics) >= 2:
            return "good"
        elif len(topics) >= 1:
            return "fair"
        else:
            return "poor"
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure."""
        return {
            "extracted_content": "",
            "content_type": "error",
            "source_analysis": {
                "original_length": 0,
                "extracted_length": 0,
                "signal_ratio": 0,
                "ray_peat_percentage": 0
            },
            "bioenergetic_content": {
                "primary_topics": [],
                "mechanisms_explained": [],
                "recommendations": [],
                "research_mentioned": []
            },
            "quality_assessment": {
                "signal_density": "low",
                "educational_value": "poor",
                "uniqueness": "basic_information",
                "completeness": "fragmentary"
            },
            "extraction_notes": reason
        }

def process_file(input_file: Path, output_dir: Path) -> Dict:
    """Process a single file with rule-based extraction."""
    extractor = SimpleRayPeatExtractor()
    
    # Read content
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract signal
    result = extractor.extract_signal(content, input_file.name)
    
    # Check if worth saving
    quality = result.get('quality_assessment', {})
    signal_density = quality.get('signal_density', 'low')
    ray_peat_pct = result.get('source_analysis', {}).get('ray_peat_percentage', 0)
    
    if signal_density in ['high', 'very_high', 'medium'] and ray_peat_pct >= 10:
        # Create enhanced content
        enhanced_content = f"""# Ray Peat Content - {input_file.stem}

## Source Information
- **Original File:** {input_file.name}
- **Content Type:** {result.get('content_type', 'unknown')}
- **Signal Ratio:** {result.get('source_analysis', {}).get('signal_ratio', 0)}%
- **Ray Peat Content:** {ray_peat_pct}%

## Quality Assessment
- **Signal Density:** {signal_density}
- **Educational Value:** {quality.get('educational_value', 'unknown')}
- **Topics Covered:** {len(result.get('bioenergetic_content', {}).get('primary_topics', []))}

## Bioenergetic Topics
{', '.join(result.get('bioenergetic_content', {}).get('primary_topics', []))}

## Key Mechanisms
{chr(10).join(f"• {mech}" for mech in result.get('bioenergetic_content', {}).get('mechanisms_explained', [])[:5])}

## Recommendations
{chr(10).join(f"• {rec}" for rec in result.get('bioenergetic_content', {}).get('recommendations', [])[:5])}

---

## Extracted Content

{result.get('extracted_content', '')}

---

## Processing Notes
{result.get('extraction_notes', 'Rule-based extraction completed')}
"""
        
        # Save enhanced file
        output_file = output_dir / f"{input_file.stem}_enhanced.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_content)
        
        return {
            "enhanced": True,
            "output_file": output_file.name,
            "signal_density": signal_density,
            "ray_peat_percentage": ray_peat_pct
        }
    
    return {
        "enhanced": False,
        "reason": f"Low quality: {signal_density}, {ray_peat_pct}% Ray Peat"
    }

def main():
    """Process Tier 1 files with rule-based enhancement."""
    input_dir = Path("../../data/processed/cleaned_corpus_tier1")
    output_dir = Path("../../data/processed/ray_peat_signal_enhanced")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Rule-Based Ray Peat Signal Enhancement")
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print()
    
    stats = {
        "files_processed": 0,
        "files_enhanced": 0,
        "high_quality": 0,
        "medium_quality": 0
    }
    
    # Process first 10 files for testing
    txt_files = list(input_dir.glob("*.txt"))[:10]
    
    for i, file_path in enumerate(txt_files, 1):
        print(f"[{i}/{len(txt_files)}] Processing: {file_path.name}")
        
        try:
            result = process_file(file_path, output_dir)
            
            if result["enhanced"]:
                stats["files_enhanced"] += 1
                if result["signal_density"] == "high":
                    stats["high_quality"] += 1
                elif result["signal_density"] == "medium":
                    stats["medium_quality"] += 1
                
                print(f"  ✅ Enhanced: {result['signal_density']} quality, {result['ray_peat_percentage']}% Ray Peat")
            else:
                print(f"  ⏭️  Skipped: {result['reason']}")
            
            stats["files_processed"] += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()
    print("✅ Rule-Based Enhancement Complete!")
    print(f"📊 Files processed: {stats['files_processed']}")
    print(f"📊 Files enhanced: {stats['files_enhanced']}")
    print(f"📊 High quality: {stats['high_quality']}")
    print(f"📊 Medium quality: {stats['medium_quality']}")
    
    if stats["files_enhanced"] > 0:
        print(f"🎉 Success! Enhanced files are in: {output_dir}")

if __name__ == "__main__":
    main() 