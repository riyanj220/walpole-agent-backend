from pathlib import Path
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import hashlib
import fitz
from tqdm import tqdm

# Chapter headings regex pattern
CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)", re.IGNORECASE)
EXAMPLE_RE = re.compile(r"^Example\s+(\d+(?:\.\d+)*):", re.IGNORECASE)

# Exercises sections regex patterns
EXERCISES_HEADING_RE = re.compile(r"^Exercises\b", re.IGNORECASE)
EXERCISE_NUMBER_RE = re.compile(r"^(\d+\.\d+)\s+(.+)")
EXERCISE_START_RE = re.compile(
    r"^(\d{1,3}\.\d{1,3})\s*(?:\([a-z]\)\s*)?[A-Za-z]", re.IGNORECASE
)
EXERCISE_SUBPART_RE = re.compile(r'^\(([a-z])\)\s+(.+)', re.IGNORECASE)

# Answers section regex patterns
ANSWERS_HEADING_RE = re.compile(
    r"^\s*Answers\s+to\s+Odd[--]?Numbered\s*$",
    re.IGNORECASE,
)
ANSWER_START_RE = re.compile(r"^(\d{1,3}\.\d{1,3})(?:\s|$|\s*\()")

# Other special sections
BLANK_PAGE_RE = re.compile(r"This page intentionally left blank", re.IGNORECASE)
MISCONCEPTIONS_RE = re.compile(r"Potential Misconceptions and Hazards", re.IGNORECASE)

# REVIEW_EXERCISES_RE = re.compile(r"Review Exercises", re.IGNORECASE)
REVIEW_EXERCISES_RE = re.compile(r"^Review\s+Exercises\s*$", re.IGNORECASE)

# Other patterns
TABLE_OR_FIGURE_RE = re.compile(r"^(Table|Figure)\s+\d+", re.IGNORECASE)
NUMERIC_ROW_RE = re.compile(r"^(\d+(\.\d+)?\s+){2,}")

def extract_detailed_text_info(pdf_path):
    """
    Extract text with detailed visual information:
    - Font size, font weight (bold), font family
    - Y-position (vertical placement)
    - Whether line is isolated (has space before/after)
    """
    doc = fitz.open(pdf_path)
    pages_info = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        lines_info = []
        
        prev_y = None
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block.get("lines", []):
                text = ""
                font_sizes = []
                font_names = []
                is_bold = False
                
                for span in line.get("spans", []):
                    text += span.get("text", "")
                    if span.get("size"):
                        font_sizes.append(span["size"])
                    if span.get("font"):
                        font_names.append(span["font"])
                        # Check if bold (font names often contain "Bold")
                        if "Bold" in span["font"] or "bold" in span["font"]:
                            is_bold = True
                
                if text.strip():
                    avg_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                    y_pos = line.get("bbox", [0,0,0,0])[1]  # Top Y coordinate
                    
                    # Check vertical spacing
                    space_before = (y_pos - prev_y) if prev_y else 0
                    
                    lines_info.append({
                        "text": text.strip(),
                        "size": avg_size,
                        "bold": is_bold,
                        "y_pos": y_pos,
                        "space_before": space_before,
                        "fonts": font_names
                    })
                    prev_y = line.get("bbox", [0,0,0,0])[3]  # Bottom Y coordinate
        
        pages_info.append(lines_info)
    
    return pages_info


def analyze_heading_patterns(pages_info, sample_pages=50):
    """
    Analyze the first N pages to learn heading patterns
    Returns thresholds for heading detection
    """
    print(" Analyzing document structure...")
    
    all_sizes = []
    heading_sizes = []
    body_sizes = []
    
    for page_idx in range(min(sample_pages, len(pages_info))):
        for line_info in pages_info[page_idx]:
            text = line_info["text"]
            size = line_info["size"]
            
            if size > 0:
                all_sizes.append(size)
                
                # Heuristic: Lines that look like headings
                if (len(text.split()) <= 12 and 
                    text[0].isupper() and 
                    not text.endswith(('.', ',', ';')) and
                    line_info["space_before"] > 10):
                    heading_sizes.append(size)
                # Likely body text
                elif len(text.split()) > 15:
                    body_sizes.append(size)
    
    # Calculate statistics
    if not all_sizes:
        return {"body_size": 10.0, "heading_threshold": 11.0, "bold_important": False}
    
    all_sizes.sort()
    body_median = sorted(body_sizes)[len(body_sizes)//2] if body_sizes else 10.0
    heading_median = sorted(heading_sizes)[len(heading_sizes)//2] if heading_sizes else 11.0
    
    # Size threshold: headings are typically 10-20% larger than body
    size_threshold = body_median + 0.5
    
    print(f"  Body text median size: {body_median:.1f}pt")
    print(f"  Heading median size: {heading_median:.1f}pt")
    print(f"  Heading threshold: {size_threshold:.1f}pt")
    
    return {
        "body_size": body_median,
        "heading_threshold": size_threshold,
        "heading_median": heading_median
    }


def is_heading_by_visual(line_info, patterns, prev_line_info=None, next_line_info=None):
    """
    Determine if line is a heading using VISUAL characteristics
    """
    text = line_info["text"].strip()
    size = line_info["size"]
    
    # Quick rejections
    if len(text) < 10 or len(text) > 150:
        return False
    
    if text.startswith(('•', '-', '*', '(', '[', 'Example', 'Table', 'Figure', 'The ', 'In ', 'It ')):
        return False
    
    if text.endswith((',', ';', ':')):
        return False
    
    words = text.split()
    if len(words) < 3 or len(words) > 15:
        return False
    
    # Visual scoring system
    score = 0
    
    # 1. Font size (MOST IMPORTANT)
    if size > patterns["heading_threshold"] + 1.5:
        score += 4
    elif size > patterns["heading_threshold"]:
        score += 3
    elif size > patterns["body_size"] + 0.3:
        score += 2
    
    # 2. Bold text
    if line_info["bold"]:
        score += 2
    
    # 3. Vertical spacing (isolated line with space before)
    if line_info["space_before"] > 15:
        score += 2
    elif line_info["space_before"] > 10:
        score += 1
    
    # 4. Capitalization
    cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
    if cap_ratio >= 0.8:
        score += 2
    elif cap_ratio >= 0.6:
        score += 1
    
    # 5. Question mark (common in textbook headings)
    if text.endswith('?'):
        score += 2
    
    # 6. Previous line context
    if prev_line_info:
        # Previous line was also isolated = strong signal
        if prev_line_info["space_before"] > 10 and len(prev_line_info["text"]) < 100:
            score += 1
    
    # 7. Next line context
    if next_line_info:
        # Next line is normal body text (longer, smaller font)
        if (len(next_line_info["text"]) > 50 and 
            next_line_info["size"] < size):
            score += 1
    
    # 8. All words capitalized
    if all(w[0].isupper() for w in words if w):
        score += 1
    
    # Threshold: need at least 5 points
    return score >= 5


def detect_and_remove_headers(pages_info, header_y_position=35.6, y_tolerance=1.0):
    """
    Remove all lines at the header Y-position (page numbers + chapter headers).
    Since headers change from page to page in textbooks, we don't need repetition checking.
    We simply remove everything at the consistent header Y-position.
    
    Args:
        pages_info: List of pages with line information
        header_y_position: The Y-position where headers appear (e.g., 35.6)
        y_tolerance: How much Y-position can vary (±2 points)
    """
    cleaned_pages = []
    removed_count = 0
    removed_examples = []
    
    for page_idx, lines in enumerate(pages_info):
        new_lines = []
        for li in lines:
            # Check if this line is at the header Y-position
            is_header_position = abs(li["y_pos"] - header_y_position) <= y_tolerance
            
            if is_header_position:
                removed_count += 1
                # Collect some examples for logging
                if len(removed_examples) < 5:
                    removed_examples.append(f"Page {page_idx+1}: '{li['text'][:60]}'")
                continue  # 🚫 Remove this line
            
            new_lines.append(li)
        cleaned_pages.append(new_lines)
    
    print(f" Removed {removed_count} header lines at y-position {header_y_position} (±{y_tolerance})")
    if removed_examples:
        print("   Examples of removed lines:")
        for ex in removed_examples:
            print(f"   {ex}")
    
    return cleaned_pages


def parse_answer_id(aid):
    """Parse answer ID like '1.1' into tuple (chapter, exercise)"""
    try:
        parts = aid.split(".")
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except (ValueError, AttributeError):
        pass
    return None


def is_valid_new_answer(new_id, prev_id):
    """Check if new_id is a valid progression from prev_id"""
    new_parsed = parse_answer_id(new_id)
    prev_parsed = parse_answer_id(prev_id) if prev_id else None
    
    if not new_parsed:
        return False
    
    new_ch, new_ex = new_parsed
    

    if new_ex % 2 == 0:
        return False
    
    # First answer is always valid (if odd)
    if not prev_parsed:
        return True
    
    prev_ch, prev_ex = prev_parsed
    
    # Valid if: same chapter and next exercise, OR next chapter and first exercise
    if new_ch == prev_ch:
        # Must be exactly +2 (next odd number) OR a reasonable skip (e.g., skipped one odd exercise)
        # Allow up to +4 increment (skipping one odd exercise: 1.1 → 1.5 is valid if 1.3 was skipped)
        diff = new_ex - prev_ex
        if diff == 2:  # Perfect: 1.1 → 1.3
            return True
        elif diff == 4:  # Some exercises progression is like 1.1 → 1.5 (skipped 1.3)
            return True
        else:
            return False  # Too large a jump or invalid progression
    
    # New chapter starting: must be first odd exercise (usually X.1)
    elif new_ch == prev_ch + 1:
        # First exercise of new chapter should be X.1 (odd)
        return new_ex == 1 or new_ex <= 5  # Allow X.1, X.3, X.5 as chapter start
    
    # Jumped multiple chapters (skipped entire chapters)
    elif new_ch > prev_ch:
        return new_ex <= 5  # Must start reasonably at beginning of new chapter
    
    return False


def structured_chunking(pages, pages_info, patterns):
    """
    Structured chunking using visual layout analysis
    """
    chunks = []
    current_text = []
    mode = "theory"
    current_meta = {"chapter": None, "section": "chapter", "type": "theory", "page": None,"answer_id": None}

    allow_chapter_start = True
    headings_detected = 0
    exercises_detected = 0
    answers_detected = 0


    def flush():
        # Flushes the current_text buffer into a new Document chunk.
        if not current_text:
            return
        text = "\n".join(current_text).strip()
        if text:
            print("="*30)
            print(f"📦 FLUSHING CHUNK")
            print(f"  Type: {current_meta.get('type')}")
            print(f"  Meta: {current_meta}")
            print(f"  Preview: {text[:100]}...{text[-100:]}")
            print("="*30)
            chunks.append(Document(page_content=text, metadata=current_meta.copy()))
        current_text.clear()

    def switch(new_mode, extra_meta=None, flush_before=True, reset_ids=True):
        nonlocal mode, current_meta
        if flush_before:
            flush()
        mode = new_mode
        if reset_ids:
            for k in ("example_id", "exercise_id", "answer_id"):
                current_meta.pop(k, None)
        if extra_meta:
            current_meta.update(extra_meta)


    print(" Processing pages with visual layout analysis...")
    
    last_exercise_num = None  
    skip_until_chapter = False


    for page_idx, page in enumerate(tqdm(pages, desc="Pages")):
        page_no = page.metadata.get("page", None)
        lines_info = pages_info[page_idx] if page_idx < len(pages_info) else []

        for line_idx, line_info in enumerate(lines_info):
            line = line_info["text"]

            # If we're inside Review Exercises section, skip all content until next Chapter
            if skip_until_chapter:
                # ...only stop skipping if we find a Chapter OR an "unlock" phrase.
                # If it's not one of these, continue skipping.
                if (not CHAPTER_RE.match(line) and 
                    not ANSWERS_HEADING_RE.match(line) and
                    not BLANK_PAGE_RE.search(line) and 
                    not MISCONCEPTIONS_RE.search(line)):
                    continue
            
            # Get prev/next line info for context
            prev_line_info = lines_info[line_idx - 1] if line_idx > 0 else None
            next_line_info = lines_info[line_idx + 1] if line_idx < len(lines_info) - 1 else None

            # Skip Review Exercises section entirely until a new Chapter starts
            if REVIEW_EXERCISES_RE.search(line):
                print(f"Skipping Review Exercises starting at page {page_no}")
                flush()
                skip_until_chapter = True
                allow_chapter_start = True
                continue

            # handle blank pages or misconceptions normally
            if BLANK_PAGE_RE.search(line) or MISCONCEPTIONS_RE.search(line):
                flush()
                current_text.append(line)
                allow_chapter_start = True
                continue
            
            if ANSWERS_HEADING_RE.match(line):
                print(f"📘 Entering Answers Section at page {page_no}")
                switch("answers_header", {
                    "section": "answers_section",
                    "type": "answers_section_header",
                    "page": page_no
                }, flush_before=True)
                current_text.append(line)
                flush()  # Flush the header as its own chunk
                mode = "answer"  # Ready to capture answers
                skip_until_chapter = False  # IMPORTANT: Don't skip in answers section!
                continue
            
            # 2. Handle content INSIDE answers section
            if mode == "answer":
                # 2a. Chapter heading inside answers section
                m_chapter = CHAPTER_RE.match(line)
                if m_chapter:
                    flush()  # Finalize previous answer
                    chapter_num = int(m_chapter.group(1))
                    current_meta['chapter'] = chapter_num
                    current_meta['page'] = page_no
                    print(f"  📖 Chapter {chapter_num} in answers section (page {page_no})")
                    # Don't append chapter line, just track it in metadata
                    continue
                
                # 2b. Individual answer detection (e.g., "1.1 (a) ..." or "2.3 ..." or "1.3" alone)
                m_answer = ANSWER_START_RE.match(line)
                if m_answer:
                    new_answer_id = m_answer.group(1)
                    prev_answer_id = current_meta.get("answer_id")
                    
                    # Parse the chapter number from the answer ID
                    parsed = parse_answer_id(new_answer_id)
                    chapter_from_id = parsed[0] if parsed else None
                    
                    # CRITICAL: Answer's chapter must match current chapter metadata
                    # We only switch chapters when we see "Chapter X" heading
                    current_chapter = current_meta.get('chapter')
                    if chapter_from_id != current_chapter:
                        # This looks like a number but belongs to different chapter
                        # It's probably a table value or page number, not an answer
                        current_text.append(line)
                        continue
                    
                    # Validate this is a real new answer
                    if is_valid_new_answer(new_answer_id, prev_answer_id):
                        flush()  # Finalize previous answer
                        answers_detected += 1
                        
                        parsed = parse_answer_id(new_answer_id)
                        chapter_from_id = parsed[0] if parsed else current_meta.get('chapter')
                        
                        print(f"Answer {new_answer_id} detected (page {page_no})")
                        
                        current_meta.update({
                            "type": "answer",
                            "answer_id": new_answer_id,
                            "chapter": chapter_from_id,
                            "section": "answers_section",
                            "page": page_no
                        })
                        current_text.append(line)
                        continue
                    else:
                        # This looks like an answer ID but doesn't progress - might be a decimal number
                        # Treat as continuation
                        current_text.append(line)
                        continue
                
                # 2c. Continuation of current answer
                current_text.append(line)
                continue
            

            m = CHAPTER_RE.match(line) 
            if m and allow_chapter_start: 
                skip_until_chapter = False
                switch( "chapter_header", {"chapter": int(m.group(1)), "section": "chapter", "type": "chapter", "page": page_no}, ) 
                current_text.append(line) 
                allow_chapter_start = False
                continue


            # Exercises section (regex)
            if EXERCISES_HEADING_RE.match(line):
                # only switch if not already in exercises_section or exercise
                if mode not in ("exercises_section", "exercise"):
                    switch("exercises_section", {
                        "section": "exercises_section",
                        "type": "exercises_section",
                        "page": page_no
                    })
                continue
            
            if mode in ("exercises_section", "exercise"):
                # Match exercise number at start: "2.31" or "2.31 Some text"
                match = EXERCISE_START_RE.match(line)
                
                if match:
                    new_ex_id = match.group(1)
                    
                    # Check if this is just a reference (not at line start)
                    if not line.startswith(new_ex_id):
                        current_text.append(line)
                        continue

                    # Check if next line starts with (a) - strong signal it's a new exercise
                    next_line = lines_info[line_idx + 1]["text"] if line_idx + 1 < len(lines_info) else ""
                    starts_with_subpart = bool(re.match(r'^\(a\)', next_line.strip()))
                    
                    # Parse numeric tuple (chapter, exercise)
                    def parse_id(eid):
                        try:
                            return tuple(map(int, eid.split(".")))
                        except Exception:
                            return None

                    new_ex_num = parse_id(new_ex_id)

                    # HAPTER CONTEXT CHECK  ---
                    current_chapter = current_meta.get('chapter')
                    if new_ex_num and current_chapter:
                        new_ch, new_ex = new_ex_num
                        # If the exercise's chapter (e.g., "47") doesn't match the current
                        # document chapter (e.g., "18"), it's a reference. Skip it.
                        if new_ch != current_chapter:
                            print(f"  -> Skipping ref {new_ex_id} (Chapter mismatch: {new_ch} != {current_chapter})")
                            current_text.append(line)
                            continue 

                    # Validation: enforce monotonic increase
                    if last_exercise_num and new_ex_num:
                        prev_ch, prev_ex = last_exercise_num
                        new_ch, new_ex = new_ex_num

                        # If number goes backward or stays same AND it's not starting with (a), 
                        # it's likely a reference
                        if (new_ch < prev_ch) or (new_ch == prev_ch and new_ex <= prev_ex):
                            if not starts_with_subpart:
                                print(f"Skipping ref {new_ex_id} (prev {last_exercise_num})")
                                current_text.append(line)
                                continue

                   # Guard: skip if numeric value looks like a decimal < 1 (probabilities, data)
                    try:
                        val = float(new_ex_id)
                        if val < 1.0:
                            # This is likely a probability or table value, not an exercise number
                            current_text.append(line)
                            continue
                    except ValueError:
                        pass

                    # NEW EXERCISE DETECTED
                    print(f"✓ New exercise {new_ex_id} (prev {last_exercise_num})")
                    if starts_with_subpart:
                        print(f"  → Starts with (a)")

                    flush()
                    exercises_detected += 1
                    last_exercise_num = new_ex_num

                    mode = "exercise"
                    current_meta.update({
                        "type": "exercise",
                        "exercise_id": new_ex_id,
                        "page": page_no,
                    })
                    current_text.append(line)
                    continue

            # Example detection (regex)
            em = EXAMPLE_RE.match(line)
            if em:
                switch(
                    "example",
                    {"section": "chapter", "type": "example", "example_id": em.group(1), "page": page_no},
                    reset_ids=False,
                )
                current_text.append(line)
                continue

            # Tables and figures
            if TABLE_OR_FIGURE_RE.match(line):
                if mode == "example" and current_meta.get("example_id"):
                    current_text.append(line)
                    continue
                switch("theory", {"section": "chapter", "type": "theory", "page": page_no})
                current_text.append(line)
                continue

            # Numeric rows (data)
            if NUMERIC_ROW_RE.match(line):
                current_text.append(line)
                continue

            # 🎯 VISUAL HEADING DETECTION
            is_heading = is_heading_by_visual(line_info, patterns, prev_line_info, next_line_info)
            
            if is_heading:
                headings_detected += 1
                print(f"  ✓ Heading: {line[:70]}...")
                # Switch to theory mode (ends any current example)
                switch("theory", {"section": "chapter", "type": "theory", "page": page_no})
                current_text.append(line)
                continue

            # Transition from chapter header to theory
            if mode == "chapter_header":
                current_meta["type"] = "theory"
                mode = "theory"

            # Default: add to current chunk
            current_text.append(line)

    flush()
    
    print(f"\nDetected {headings_detected} theory headings")
    print(f"Detected {exercises_detected} individual exercises")
    
    return chunks


def normalize_text(text: str) -> str:
    """Normalize text for deduplication"""
    return " ".join(text.lower().split())


def deduplicate_chunks(chunks, debug=True, safety_check=True):
    """
    Remove duplicate chunks based on content and metadata.
    
    Args:
        chunks: List of Document objects
        debug: Print detailed info about removed duplicates
        safety_check: Only remove if content is EXACTLY the same
    """
    seen = {}  # Map hash -> first occurrence
    unique = []
    duplicates_removed = 0
    
    for idx, doc in enumerate(chunks):
        norm = normalize_text(doc.page_content)
        
        # Create metadata signature
        meta_key = (
            str(doc.metadata.get("type")) +
            str(doc.metadata.get("exercise_id", "")) +
            str(doc.metadata.get("example_id", "")) +
            str(doc.metadata.get("answer_id", "")) +
            str(doc.metadata.get("chapter", ""))
        )
        
        # Hash content + metadata
        h = hashlib.md5((norm + meta_key).encode("utf-8")).hexdigest()

        if h not in seen:
            seen[h] = idx
            unique.append(doc)
        else:
            duplicates_removed += 1
            if debug:
                first_idx = seen[h]
                print(f"   Duplicate #{idx} (same as #{first_idx})")
                print(f"   Type: {doc.metadata.get('type')}")
                print(f"   ID: {doc.metadata.get('exercise_id') or doc.metadata.get('answer_id') or 'N/A'}")
                print(f"   Preview: {doc.page_content[:60]}...")
    
    if duplicates_removed > 0:
        print(f"\n📊 Deduplication: Removed {duplicates_removed} duplicates from {len(chunks)} chunks")
        print(f"   Final count: {len(unique)} unique chunks")
    else:
        print(f"No duplicates found in {len(chunks)} chunks")
    
    return unique


def filter_pages(docs, pages_info, skip_ranges):
    """
    Removes pages that fall within any of the given skip_ranges.
    
    Args:
        docs (list): List of Document objects from PyPDFLoader.
        pages_info (list): List of detailed page layout info.
        skip_ranges (list): A list of tuples, e.g., [(start_page, end_page)].
        
    Returns:
        A tuple containing (filtered_docs, filtered_pages_info).
    """
    filtered_docs = []
    filtered_pages_info = []

    for doc, page_layout in zip(docs, pages_info):
        # PyPDFLoader page numbers are 1-based by default
        page_num = doc.metadata.get("page", 0) 
        
        # Check if the current page falls into any of the ranges to be skipped
        is_skipped = any(start <= page_num <= end for start, end in skip_ranges)
        
        if not is_skipped:
            filtered_docs.append(doc)
            filtered_pages_info.append(page_layout)
            
    return filtered_docs, filtered_pages_info


def ingest_pdf():
    """Main ingestion pipeline with visual layout analysis"""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    pdf_path = data_dir / "walpole.pdf"
    out_path = data_dir / "walpole"

    print("Loading PDF...")
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    docs = docs[21:805]

    pages_info = extract_detailed_text_info(str(pdf_path))
    pages_info = pages_info[21:805]


    # NEW: Define page ranges to exclude (e.g., bibliography, index)
    skip_ranges = [
        (741, 788) 
    ]
    
    print(f"exclusionary filtering pages from ranges: {skip_ranges}...")
    initial_page_count = len(docs)
    docs, pages_info = filter_pages(docs, pages_info, skip_ranges)
    print(f"Removed {initial_page_count - len(docs)} pages. Proceeding with {len(docs)} pages.")

    print("Removing headers/footers...")
    pages_info = detect_and_remove_headers(
        pages_info, 
        header_y_position=35.6,
        y_tolerance=1.0
    )

    print("Learning document structure patterns...")
    patterns = analyze_heading_patterns(pages_info, sample_pages=100)

    print("Performing structured chunking...")
    structured_docs = structured_chunking(docs, pages_info, patterns)

    print(f"Created {len(structured_docs)} structured chunks")

    print("Splitting large theory and example chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    final_chunks = []

    for doc in structured_docs:
        if doc.metadata.get("type") in ["theory", "example"] and len(doc.page_content) > 800:
            final_chunks.extend(splitter.split_documents([doc]))
        else:
            final_chunks.append(doc)

    print("Deduplicating chunks...")
    final_chunks = deduplicate_chunks(final_chunks, debug=True)

    print(f"Final chunk count: {len(final_chunks)}")
    
    # Print statistics
    type_counts = {}
    for chunk in final_chunks:
        chunk_type = chunk.metadata.get("type", "unknown")
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
    
    print("\n📊 Chunk Statistics:")
    for chunk_type, count in sorted(type_counts.items()):
        print(f"  {chunk_type}: {count}")

    print("\n🧮 Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vectorstore = FAISS.from_documents(final_chunks, embeddings)
    vectorstore.save_local(str(out_path))
    
    print(f"\nSaved vector store to {out_path}")
    print("Done!")


if __name__ == "__main__":
    ingest_pdf()