# Ray Peat Anthology Processing Pipeline

## Overview

This pipeline analyzes the Ray Peat Anthology Excel file and processes new content that hasn't been incorporated into the existing dataset. The goal is to scrape, process, and archive the remaining data to expand the Ray Peat corpus.

## Files in this Directory

### Analysis Scripts
- **`analyze_anthology.py`** - Initial comprehensive analysis of the Excel file vs existing data
- **`extract_urls_for_processing.py`** - Extracts and prioritizes URLs for processing
- **`anthology_summary.py`** - Provides summary and recommendations
- **`web_scraper.py`** - Main scraping tool for downloading content

### Output Files
- **`anthology_analysis.json`** - Detailed analysis results (large file)
- **`urls_for_processing.json`** - Prioritized URLs ready for processing

## Key Findings

### Summary Statistics
- **Total URLs in anthology**: 2,603
- **After duplicate filtering**: 1,731 URLs remain
- **High priority URLs**: 659 (Ray Peat official content + key domains)
- **Estimated processing time**: ~87 hours (with rate limiting)

### Content Breakdown
| Content Type | Count | Description |
|--------------|-------|-------------|
| Video | 657 | YouTube interviews/presentations |
| Audio | 475 | Podcasts, radio shows |
| PDF | 303 | Academic papers, documents |
| Other | 279 | Articles, web content |
| Article | 17 | Specific articles |

### Top Domains
| Domain | URL Count | Priority |
|--------|-----------|----------|
| youtube.com | 627 | High (videos with transcripts) |
| www.functionalps.com | 257 | Medium |
| www.toxinless.com | 236 | Medium |
| wiki.chadnet.org | 196 | Medium |
| raypeat.com | 2 | **Highest** (official content) |
| raypeatforum.com | 141 | Medium |

## Processing Strategy

### Phase 1: High Priority (659 URLs)
Focus on:
- **raypeat.com** - Official Ray Peat content (highest priority)
- **YouTube videos** - Interviews and presentations with transcripts
- **Key interview sources** - Well-known hosts/platforms

### Phase 2: Medium Priority (1 URL)
- Specialized content requiring manual review

### Phase 3: Low Priority (1,071 URLs)
- Remaining content for comprehensive coverage

## Usage Instructions

### Prerequisites
```bash
# Activate virtual environment
source ../../venv/bin/activate

# Required packages are already installed:
# pandas, openpyxl, xlrd, requests, beautifulsoup4, yt-dlp
```

### 1. Analysis and Planning
```bash
# Generate fresh URL analysis
python extract_urls_for_processing.py

# View summary and recommendations
python anthology_summary.py
```

### 2. Content Processing

#### Test Run (Recommended First Step)
```bash
# Test with 5 high-priority URLs
python web_scraper.py --priority high --max-urls 5
```

#### Production Processing
```bash
# Process all high-priority URLs (659 URLs)
python web_scraper.py --priority high

# Process in smaller batches
python web_scraper.py --priority high --max-urls 50

# Process specific priority levels
python web_scraper.py --priority medium
python web_scraper.py --priority low
```

### 3. Quality Control
After processing batches:
1. Check output in `../../data/raw/raw_data/`
2. Review logs for failed URLs
3. Manually inspect content quality
4. Adjust duplicate filtering if needed

## Output Organization

The scraper automatically organizes content into appropriate folders:

```
data/raw/raw_data/
├── 01_Audio_Transcripts/
│   ├── Video_Content/          # YouTube videos with transcripts
│   └── Audio_Content/          # Audio files and transcripts
├── 02_Publications/
│   ├── PDF_Documents/          # PDF papers and documents
│   └── Articles/               # Web articles and blog posts
└── 09_Miscellaneous/          # Other content types
```

## Technical Features

### Rate Limiting
- 2-second delay between requests
- Respectful scraping practices
- Session management for efficiency

### Content Extraction
- **Web pages**: BeautifulSoup for clean text extraction
- **YouTube videos**: yt-dlp for transcript extraction
- **PDFs**: Direct download (manual processing needed)

### Duplicate Detection
- Title-based matching (70% overlap threshold)
- Filename comparison with existing corpus
- Adjustable filtering parameters

### Error Handling
- Graceful failure for inaccessible URLs
- Detailed logging and statistics
- Resume capability for interrupted processing

## Monitoring and Statistics

The scraper provides real-time statistics:
- **Processed**: Total URLs attempted
- **Successful**: Successfully downloaded and saved
- **Failed**: URLs that couldn't be processed
- **Progress**: Current position in batch

## Troubleshooting

### Common Issues

1. **YouTube transcript unavailable**
   - Some videos don't have auto-generated or manual transcripts
   - Consider manual transcript creation for high-value content

2. **Rate limiting errors**
   - Increase delay between requests
   - Check domain-specific rate limits

3. **File naming conflicts**
   - Scraper automatically handles name sanitization
   - Duplicates get timestamped suffixes

### Manual Processing Needed

Some content types require manual intervention:
- **Password-protected sites**
- **Content behind paywalls**
- **Interactive media**
- **PDF processing** (currently downloads but needs text extraction)

## Next Steps for Production

1. **Start with test batch** (5-10 URLs)
2. **Review content quality**
3. **Process high-priority batch** (659 URLs)
4. **Integrate with existing AI processing pipeline**
5. **Scale to medium/low priority content**

## Integration with Existing Pipeline

After scraping, new content should flow through:
1. **Raw content** → `data/raw/raw_data/`
2. **AI cleaning** → `preprocessing/cleaning/ai_powered_cleaners.py`
3. **Processed content** → `data/processed/ai_cleaned/`
4. **Embedding** → `embedding/embed_corpus.py`

## Performance Expectations

- **Processing speed**: ~3 seconds per URL (with rate limiting)
- **Success rate**: ~70-80% (based on test runs)
- **Storage requirements**: ~10-50KB per text file, varies for PDFs

## Contact and Support

For issues or improvements:
- Check logs in console output
- Review failed URLs for patterns
- Adjust filtering parameters in `extract_urls_for_processing.py`
- Modify scraping logic in `web_scraper.py` for specific domains 