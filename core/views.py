from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from .rag_pipeline import (
    ask_pipeline, 
    batch_ask, 
    get_chapter_summary,
    health_check
)
import logging

logger = logging.getLogger(__name__)


@api_view(["POST"])
def ask(request):
    """
    Main query endpoint - Ask a question to the RAG system.
    
    POST body:
    {
        "query": "What is exercise 6.14?",
        "params": {
            "chapter": 6,           // Optional: filter by chapter
            "mode": "agent",        // Optional: force 'agent' or 'direct' mode
            "max_results": 5        // Optional: max results to return
        }
    }
    
    Response:
    {
        "answer": "...",
        "mode": "agent|direct",
        "metadata": {...},
        "query": "..."
    }
    """
    try:
        payload = request.data or {}
        query = payload.get("query", "").strip()
        params = payload.get("params", {})

        if not query:
            return JsonResponse(
                {"error": "query is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"Processing query: {query[:100]}")
        
        # Process query through pipeline
        result = ask_pipeline(query, params=params)
        
        # Add original query to response
        result["query"] = query
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return JsonResponse(
            {"error": f"Internal server error: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["POST"])
def batch_query(request):
    """
    Batch query endpoint - Process multiple queries at once.
    
    POST body:
    {
        "queries": ["What is exercise 6.14?", "Explain variance"],
        "params": {
            "chapter": 6
        }
    }
    
    Response:
    {
        "results": [
            {"answer": "...", "mode": "...", ...},
            {"answer": "...", "mode": "...", ...}
        ],
        "total": 2
    }
    """
    try:
        payload = request.data or {}
        queries = payload.get("queries", [])
        params = payload.get("params", {})

        if not queries or not isinstance(queries, list):
            return JsonResponse(
                {"error": "queries must be a non-empty list"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"Processing {len(queries)} queries in batch")
        
        # Process all queries
        results = batch_ask(queries, params=params)
        
        return Response({
            "results": results,
            "total": len(results)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error processing batch query: {str(e)}", exc_info=True)
        return JsonResponse(
            {"error": f"Internal server error: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["GET"])
def chapter_info(request, chapter_number):
    """
    Get information about a specific chapter.
    
    GET /api/chapter/{chapter_number}
    
    Response:
    {
        "chapter": 6,
        "summary": {
            "total_chunks": 150,
            "exercises": 30,
            "examples": 10,
            "theory_sections": 100,
            "answers": 15
        },
        "exercise_ids": ["6.1", "6.3", ...],
        "example_ids": ["6.1", "6.2", ...],
        "available_answers": ["6.1", "6.3", ...]
    }
    """
    try:
        chapter = int(chapter_number)
        summary = get_chapter_summary(chapter)
        return Response(summary, status=status.HTTP_200_OK)
        
    except ValueError:
        return JsonResponse(
            {"error": "Invalid chapter number"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        logger.error(f"Error getting chapter info: {str(e)}", exc_info=True)
        return JsonResponse(
            {"error": f"Internal server error: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(["GET"])
def system_health(request):
    """
    Health check endpoint - Check if RAG system is working.
    
    GET /api/health
    
    Response:
    {
        "status": "healthy",
        "total_documents": 5000,
        "document_types": {
            "exercise": 500,
            "answer": 250,
            "example": 200,
            "theory": 4050
        },
        "vectorstore": "loaded"
    }
    """
    try:
        health = health_check()
        
        response_status = (
            status.HTTP_200_OK if health["status"] == "healthy" 
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        
        return Response(health, status=response_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JsonResponse(
            {
                "status": "error",
                "error": str(e)
            }, 
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@api_view(["GET"])
def list_chapters(request):
    """
    List all available chapters in the system.
    
    GET /api/chapters
    
    Response:
    {
        "chapters": [1, 2, 3, ...],
        "total": 20
    }
    """
    try:
        from .rag_runtime import vectorstore
        
        # Get all unique chapter numbers
        chapters = set()
        for doc in vectorstore.docstore._dict.values():
            chapter = doc.metadata.get('chapter')
            if chapter:
                chapters.add(chapter)
        
        chapters_list = sorted(list(chapters))
        
        return Response({
            "chapters": chapters_list,
            "total": len(chapters_list)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error listing chapters: {str(e)}", exc_info=True)
        return JsonResponse(
            {"error": f"Internal server error: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )