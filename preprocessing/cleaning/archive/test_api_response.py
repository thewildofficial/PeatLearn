#!/usr/bin/env python3
"""
Test script to debug API response issues.
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Simple test prompt
TEST_PROMPT = """
You are an expert at extracting Ray Peat's bioenergetic wisdom.

Extract pure Ray Peat signal from this content and return a JSON object with:
{
  "extracted_content": "The cleaned content",
  "signal_quality": "high",
  "ray_peat_percentage": 50
}

Content: This is a test about thyroid hormone and metabolism by Ray Peat.
"""

def test_api():
    """Test basic API functionality."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        print("🔧 Testing API response...")
        
        response = model.generate_content(
            TEST_PROMPT,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )
        
        print("📝 Raw response:")
        print(repr(response.text))
        print()
        print("📄 Formatted response:")
        print(response.text)
        
        # Try to parse JSON
        try:
            # Simple JSON extraction
            text = response.text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                parts = text.split('```')
                for part in parts:
                    if part.strip().startswith('{'):
                        text = part
                        break
            
            # Find JSON boundaries
            start_idx = text.find('{')
            if start_idx == -1:
                print("❌ No JSON object found")
                return
            
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            json_str = text[start_idx:end_idx]
            print("🔍 Extracted JSON string:")
            print(repr(json_str))
            
            result = json.loads(json_str)
            print("✅ Successfully parsed JSON:")
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
    
    except Exception as e:
        print(f"❌ API error: {e}")

if __name__ == "__main__":
    test_api() 