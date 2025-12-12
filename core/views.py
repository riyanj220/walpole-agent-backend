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
from .rag_runtime import supabase  # <--- Import Supabase client
import logging

logger = logging.getLogger(__name__)

# =====================================================
#  HELPER: Fetch Chat History
# =====================================================
def get_chat_history(chat_id, limit=6):
    """
    Fetches the last 'limit' messages for context.
    Returns a formatted string or list for the LLM.
    """
    if not supabase or not chat_id:
        return []

    try:
        # Fetch messages (descending to get newest, then reverse)
        response = supabase.table("messages")\
            .select("role, content")\
            .eq("chat_id", chat_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        data = response.data
        if not data:
            return []

        # Reverse to chronological order (Oldest -> Newest)
        history_chronological = data[::-1]
        
        # Format as list of tuples or dicts for your pipeline
        # Format: [("user", "msg"), ("assistant", "msg")]
        formatted_history = [
            (msg['role'], msg['content']) 
            for msg in history_chronological
        ]
        return formatted_history

    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return []

@api_view(["POST"])
def ask(request):
    """
    Main query endpoint - Ask a question to the RAG system.
    
    POST body:
    {
        "query": "What is exercise 6.14?",
        "user_id": "uuid-from-frontend",    # <--- NEW
        "chat_id": "uuid-or-null",          # <--- NEW
        "params": {
            "chapter": 6, 
            "mode": "agent", 
            "max_results": 5
        }
    }
    """
    try:
        payload = request.data or {}
        query = payload.get("query", "").strip()
        user_id = payload.get("user_id") # Get user_id from frontend
        chat_id = payload.get("chat_id") # Get chat_id from frontend (if active)
        params = payload.get("params", {})

        if not query:
            return JsonResponse(
                {"error": "query is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        logger.info(f"Processing query: {query[:100]} | User: {user_id} | Chat: {chat_id}")
        
        # =====================================================
        #  1. DATABASE: HANDLE CHAT SESSION
        # =====================================================
        # If we have a user but no chat_id, this is a NEW conversation.
        if supabase and user_id and not chat_id:
            try:
                # Generate a simple title from the first few words
                title = " ".join(query.split()[:5]) + "..."
                
                # Insert new chat row
                chat_data = supabase.table("chats").insert({
                    "user_id": user_id,
                    "title": title
                }).execute()
                
                # Grab the new ID
                if chat_data.data:
                    chat_id = chat_data.data[0]['id']
                    logger.info(f"Created new chat session: {chat_id}")
            except Exception as e:
                logger.error(f"Failed to create chat session: {e}")
                # We continue anyway, just without saving history
        
        # =====================================================
        #  2. DATABASE: SAVE USER MESSAGE
        # =====================================================
        if supabase and chat_id:
            try:
                supabase.table("messages").insert({
                    "chat_id": chat_id,
                    "role": "user",
                    "content": query
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save user message: {e}")

        # =====================================================
        #  3. FETCH HISTORY FOR CONTEXT
        # =====================================================
        chat_history = []
        if chat_id:
            chat_history = get_chat_history(chat_id, limit=6)
            
            # Remove the very last message if it matches our current query (since we just inserted it)
            if chat_history and chat_history[-1][1] == query:
                chat_history.pop()

        # =====================================================
        #  4. CORE PIPELINE (Pass history)
        # =====================================================
        # Pass chat_history to the pipeline
        result = ask_pipeline(query, params=params, chat_history=chat_history)
        
        result["query"] = query
        result["chat_id"] = chat_id 

        # 5. DATABASE: SAVE BOT RESPONSE
        if supabase and chat_id:
            try:
                bot_content = result.get("answer", "")
                supabase.table("messages").insert({
                    "chat_id": chat_id,
                    "role": "assistant",
                    "content": bot_content
                }).execute()
            except Exception as e:
                logger.error(f"Failed to save bot response: {e}")
        
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