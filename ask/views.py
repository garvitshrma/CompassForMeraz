"""
 ask/views.py — two endpoints. That is the whole HTTP surface.

   GET  /health   what is loaded, for debugging and uptime monitors
   POST /ask      the question endpoint

Notice how thin /ask is. All the real work is in core/, which imports no Django
and can be run from a plain script — the same separation that let the parent
project swap its entire web framework without touching its pipeline.
"""

import logging

from rest_framework.decorators import api_view
from rest_framework.exceptions import APIException
from rest_framework.response import Response

from core.answer import answer as answer_question
from core.answer import canonicalize_result

from compass.boot import warm
from ask.serializers import AskInSerializer, AskOutSerializer
from ask.state import STATE, cache_get, cache_put

log = logging.getLogger()


class HttpError(APIException):
    """Stand-in for FastAPI's HTTPException: takes the status code as an
    argument, so the body shape stays {"detail": "..."} for any code."""

    def __init__(self, status_code, detail):
        self.status_code = status_code
        super().__init__(detail)


@api_view(["GET"])
def health(request):
    """Every deployed service needs one. Also the fastest way to answer "why
    does the bot say it doesn't know anything" — if chunks is 0, nothing was
    ingested, and that is almost always the answer."""
    warm()
    return Response({
        "status": "ok",
        "chunks": len(STATE["chunks"]),
        "title": STATE["chunks"][0]["page_title"] if STATE["chunks"] else None,
    })


@api_view(["POST"])
def ask(request):
    warm()

    s = AskInSerializer(data=request.data)
    s.is_valid(raise_exception=True)
    question = s.validated_data["question"]

    if not question.strip():
        raise HttpError(400, "question required")

    # Cache lookup before any work at all. High hit rate on an FAQ bot, and a
    # hit costs no time and no tokens.
    hit = cache_get(question)
    if hit is not None:
        return Response(AskOutSerializer(canonicalize_result(hit)).data)

    result = answer_question(question, STATE["chunks"], STATE["vecs"],
                             STATE["llm"])

    # Only cache real verdicts. A transient failure — Groq down, rate-limited,
    # timed out — must NOT be frozen in for the next 500 queries; that would
    # turn a thirty-second blip into a lasting outage for exactly the questions
    # people ask most. Genuine refusals ARE cached: "we have nothing about
    # parking" is a stable fact about the index, not a transient error.
    if result.get("reason") != "the assistant is temporarily unavailable":
        cache_put(question, result)

    # Links are rewritten to the canonical origin LAST, on the way out, so
    # both a fresh answer and a cached one carry the right host — and so the
    # cache keeps whatever the index held, without the rewrite baked in.
    return Response(AskOutSerializer(canonicalize_result(result)).data)
