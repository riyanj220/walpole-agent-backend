from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from .rag_pipeline import ask_pipeline

@api_view(["POST"])
def ask(request):
    payload = request.data or {}
    q = payload.get("query", "")
    params = payload.get("params", {})

    if not q:
        return JsonResponse({"error": "query is required"}, status=400)
    return Response(ask_pipeline(q, params=params))
