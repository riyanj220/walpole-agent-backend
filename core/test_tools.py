"""
Direct testing of RAG tools without API layer.
Run this to verify your tools work correctly.
"""

# # Add your Django project to path if needed
# import sys
# import os
# import django

# # Setup Django environment
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
# django.setup()

# Now import your tools
from core.rag_tools import (
    get_exercise,
    get_answer,
    get_examples,
    get_theory_concepts,
    explain_with_context,
    smart_search
)
from core.rag_runtime import vectorstore


def print_section(title):
    """Pretty print section headers"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_vectorstore():
    """Test if vectorstore is loaded correctly"""
    print_section("TEST 1: Vectorstore Status")
    
    try:
        total_docs = len(vectorstore.docstore._dict)
        print(f"✅ Vectorstore loaded successfully!")
        print(f"📊 Total documents: {total_docs}")
        
        # Count by type
        type_counts = {}
        for doc in vectorstore.docstore._dict.values():
            doc_type = doc.metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        print("\n📚 Document breakdown:")
        for doc_type, count in sorted(type_counts.items()):
            print(f"   - {doc_type}: {count}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        return False


def test_get_exercise():
    """Test exercise retrieval"""
    print_section("TEST 2: Get Exercise")
    
    # Try a few exercise IDs
    test_ids = ["1.1", "2.5", "6.14"]
    
    for ex_id in test_ids:
        print(f"🔍 Searching for Exercise {ex_id}...")
        docs = get_exercise(ex_id)
        
        if docs:
            print(f"✅ Found {len(docs)} chunk(s)")
            print(f"📄 Preview: {docs[0].page_content[:150]}...")
            print(f"📌 Metadata: {docs[0].metadata}")
        else:
            print(f"❌ Exercise {ex_id} not found")
        print()


def test_get_answer():
    """Test answer retrieval"""
    print_section("TEST 3: Get Answer")
    
    # Test with odd-numbered exercises (answers are only for odd)
    test_ids = ["1.1", "1.3", "2.1"]
    
    for ex_id in test_ids:
        print(f"🔍 Searching for Answer to {ex_id}...")
        docs = get_answer(ex_id)
        
        if docs:
            print(f"✅ Found {len(docs)} answer chunk(s)")
            print(f"📄 Preview: {docs[0].page_content[:150]}...")
            print(f"📌 Metadata: {docs[0].metadata}")
        else:
            print(f"❌ Answer to {ex_id} not found")
        print()


def test_get_examples():
    """Test example retrieval"""
    print_section("TEST 4: Get Examples")
    
    # Test specific example
    print("🔍 Searching for Example 1.1...")
    docs = get_examples(example_id="1.1")
    
    if docs:
        print(f"✅ Found {len(docs)} example chunk(s)")
        print(f"📄 Preview: {docs[0].page_content[:150]}...")
        print(f"📌 Metadata: {docs[0].metadata}")
    else:
        print("❌ Example 1.1 not found")
    
    print("\n" + "-"*70 + "\n")
    
    # Test chapter examples
    print("🔍 Getting all examples from Chapter 1...")
    docs = get_examples(chapter=1, limit=3)
    
    if docs:
        print(f"✅ Found {len(docs)} examples in Chapter 1")
        for i, doc in enumerate(docs, 1):
            ex_id = doc.metadata.get('example_id', 'unknown')
            print(f"   {i}. Example {ex_id}")
    else:
        print("❌ No examples found in Chapter 1")


def test_get_theory():
    """Test theory/conceptual retrieval"""
    print_section("TEST 5: Get Theory Concepts")
    
    queries = [
        "variance",
        "probability distribution",
        "Bayes theorem"
    ]
    
    for query in queries:
        print(f"🔍 Searching for: '{query}'")
        docs = get_theory_concepts(query, limit=3)
        
        if docs:
            print(f"✅ Found {len(docs)} relevant chunks")
            print(f"📄 Top result preview: {docs[0].page_content[:120]}...")
            print(f"📌 Chapter: {docs[0].metadata.get('chapter', 'unknown')}")
        else:
            print(f"❌ No theory found for '{query}'")
        print()


def test_smart_search():
    """Test smart search functionality"""
    print_section("TEST 6: Smart Search")
    
    test_queries = [
        "exercise 1.1",
        "answer to 1.3",
        "example 2.1",
        "what is variance"
    ]
    
    for query in test_queries:
        print(f"🔍 Query: '{query}'")
        result = smart_search(query)
        
        print(f"   Type detected: {result['type']}")
        print(f"   Results found: {len(result['results'])}")
        if result['results']:
            print(f"   Preview: {result['formatted_text'][:100]}...")
        print()


def test_explain_with_context():
    """Test explain with context (uses LLM)"""
    print_section("TEST 7: Explain with Context (LLM)")
    
    print("⚠️  This test uses the LLM and may take a few seconds...")
    print()
    
    query = "What is variance and how is it used?"
    print(f"🔍 Query: '{query}'")
    print("\n⏳ Generating explanation...")
    
    try:
        explanation = explain_with_context(query, chapter=3)
        print("\n✅ Explanation generated:")
        print("-" * 70)
        print(explanation)
        print("-" * 70)
    except Exception as e:
        print(f"\n❌ Error generating explanation: {e}")


def test_chapter_coverage():
    """Test what chapters are available"""
    print_section("TEST 8: Chapter Coverage")
    
    chapters = set()
    for doc in vectorstore.docstore._dict.values():
        chapter = doc.metadata.get('chapter')
        if chapter:
            chapters.add(chapter)
    
    chapters_list = sorted(list(chapters))
    print(f"📚 Available chapters: {len(chapters_list)}")
    print(f"   Chapters: {chapters_list}")
    
    # Test a specific chapter
    test_chapter = chapters_list[0] if chapters_list else 1
    print(f"\n🔍 Analyzing Chapter {test_chapter}...")
    
    chapter_docs = [
        d for d in vectorstore.docstore._dict.values()
        if d.metadata.get('chapter') == test_chapter
    ]
    
    type_counts = {}
    for doc in chapter_docs:
        doc_type = doc.metadata.get('type', 'unknown')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    print(f"   Total chunks: {len(chapter_docs)}")
    print(f"   Breakdown:")
    for doc_type, count in sorted(type_counts.items()):
        print(f"      - {doc_type}: {count}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀 "*20)
    print("Starting RAG Tools Direct Testing")
    print("🚀 "*20)
    
    # Test 1: Vectorstore
    if not test_vectorstore():
        print("\n❌ Vectorstore failed to load. Cannot continue tests.")
        return
    
    # Test 2-8: Individual tools
    test_get_exercise()
    test_get_answer()
    test_get_examples()
    test_get_theory()
    test_smart_search()
    test_chapter_coverage()
    
    # Test 7: LLM test (optional, takes time)
    print("\n" + "="*70)
    response = input("Do you want to test LLM explanation? (y/n): ")
    if response.lower() == 'y':
        test_explain_with_context()
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()