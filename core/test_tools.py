"""
Diagnostic script to test your RAG system.
Run this to identify issues before deploying.
"""

from . import rag_runtime
from . import rag_tools 
from . import rag_agent

def test_vectorstore():
    """Test 1: Check vectorstore is loaded correctly"""
    print("\n" + "="*60)
    print("TEST 1: VECTORSTORE HEALTH CHECK")
    print("="*60)
    
    stats = rag_runtime.get_stats()
    print(f"✓ Total documents: {stats['total_documents']}")
    print(f"✓ Document types: {stats['by_type']}")
    print(f"✓ Chapter count: {len(stats['by_chapter'])} chapters")
    print(f"✓ Sample exercise IDs: {stats['sample_exercise_ids'][:5]}")
    print(f"✓ Sample answer IDs: {stats['sample_answer_ids'][:5]}")
    
    return stats


def test_specific_exercises():
    """Test 2: Check if specific exercises exist"""
    print("\n" + "="*60)
    print("TEST 2: SPECIFIC EXERCISE LOOKUP")
    print("="*60)
    
    test_ids = ["6.13", "6.14", "1.1", "2.5"]
    
    for ex_id in test_ids:
        result = rag_runtime.check_exercise_exists(ex_id)
        status = "✓ FOUND" if result['found'] else "✗ NOT FOUND"
        print(f"{status} - Exercise {ex_id}: {result['count']} chunks")
        
        if result['found']:
            print(f"    Metadata: {result['metadata'][0]}")


def test_specific_answers():
    """Test 3: Check if answers exist"""
    print("\n" + "="*60)
    print("TEST 3: SPECIFIC ANSWER LOOKUP")
    print("="*60)
    
    test_ids = ["6.13", "6.14", "1.1", "2.5"]
    
    for ex_id in test_ids:
        result = rag_runtime.check_answer_exists(ex_id)
        status = "✓ FOUND" if result['found'] else "✗ NOT FOUND"
        print(f"{status} - Answer {ex_id}: {result['count']} chunks")
        
        if result['found']:
            print(f"    Metadata: {result['metadata'][0]}")


def test_retrieval_functions():
    """Test 4: Test retrieval functions"""
    print("\n" + "="*60)
    print("TEST 4: RETRIEVAL FUNCTION TESTS")
    print("="*60)
    
    # Test exercise retrieval
    print("\n4a) Getting exercise 6.13:")
    docs = rag_tools.get_exercise("6.13")
    if docs:
        print(f"✓ Found {len(docs)} chunks")
        print(f"  Preview: {docs[0].page_content[:150]}...")
    else:
        print("✗ Not found")
    
    # Test answer retrieval
    print("\n4b) Getting answer 6.13:")
    docs = rag_tools.get_answer("6.13")
    if docs:
        print(f"✓ Found {len(docs)} chunks")
        print(f"  Preview: {docs[0].page_content[:150]}...")
    else:
        print("✗ Not found")
    
    # Test theory search
    print("\n4c) Searching theory for 'normal distribution':")
    docs = rag_tools.get_theory_concepts("normal distribution", chapter=6)
    if docs:
        print(f"✓ Found {len(docs)} chunks")
        print(f"  Preview: {docs[0].page_content[:150]}...")
    else:
        print("✗ Not found")


def test_direct_mode():
    """Test 5: Test direct mode queries"""
    print("\n" + "="*60)
    print("TEST 5: DIRECT MODE QUERIES")
    print("="*60)
    
    test_queries = [
        ("What is exercise 6.13?", None),
        ("Solve exercise 6.13", None),
        ("What is the answer to 6.13?", None),
        ("Explain normal distribution", 6)
    ]
    
    for query, chapter in test_queries:
        print(f"\n5.{test_queries.index((query, chapter)) + 1}) Query: '{query}'")
        result = rag_agent.ask_direct(query, chapter)
        print(f"Type: {result.get('type')}")
        print(f"Success: {result['metadata'].get('success')}")
        print(f"Results: {result['metadata'].get('num_results')}")
        print(f"Answer preview: {result['answer'][:200]}...")


def run_all_tests():
    """Run all diagnostic tests"""
    print("\n" + "="*60)
    print("WALPOLE RAG SYSTEM DIAGNOSTICS")
    print("="*60)
    
    try:
        test_vectorstore()
        test_specific_exercises()
        test_specific_answers()
        test_retrieval_functions()
        test_direct_mode()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()