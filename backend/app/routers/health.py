"""
NyayaMitra — Health Check Router.

Provides endpoints to verify that all infrastructure services
are running and accessible. Used by:
- Docker health checks
- Kubernetes liveness/readiness probes
- Monitoring (Prometheus/Grafana)
- Manual debugging (make status)

Endpoints:
    GET /api/v1/health         Quick health check (just returns OK)
    GET /api/v1/health/detail  Detailed check of every service
"""

import time
from typing import Any

import structlog
from fastapi import APIRouter

from app.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/health")


@router.get("")
async def health_check():
    """
    Quick health check — returns 200 if the API is running.

    This is the endpoint Docker and Kubernetes call for liveness probes.
    It does NOT check downstream services (use /detail for that).
    """
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
    }


@router.get("/detail")
async def health_check_detail():
    """
    Detailed health check — tests connectivity to every infrastructure service.

    Returns individual status for: PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j.
    Overall status is 'healthy' only if ALL services are reachable.
    """
    checks: dict[str, Any] = {}
    overall_healthy = True

    # ─── PostgreSQL ──────────────────────────────────────────────────────
    checks["postgresql"] = await _check_postgres()
    if checks["postgresql"]["status"] != "healthy":
        overall_healthy = False

    # ─── Redis ───────────────────────────────────────────────────────────
    checks["redis"] = await _check_redis()
    if checks["redis"]["status"] != "healthy":
        overall_healthy = False

    # ─── Qdrant ──────────────────────────────────────────────────────────
    checks["qdrant"] = await _check_qdrant()
    if checks["qdrant"]["status"] != "healthy":
        overall_healthy = False

    # ─── Elasticsearch ───────────────────────────────────────────────────
    checks["elasticsearch"] = await _check_elasticsearch()
    if checks["elasticsearch"]["status"] != "healthy":
        overall_healthy = False

    # ─── Neo4j ───────────────────────────────────────────────────────────
    checks["neo4j"] = await _check_neo4j()
    if checks["neo4j"]["status"] != "healthy":
        overall_healthy = False

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "services": checks,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Service Checks
# ═══════════════════════════════════════════════════════════════════════════════


async def _check_postgres() -> dict[str, Any]:
    """Test PostgreSQL connectivity."""
    start = time.time()
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB,
            timeout=5,
        )
        version = await conn.fetchval("SELECT version()")
        await conn.close()
        return {
            "status": "healthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "version": version.split(",")[0] if version else "unknown",
        }
    except Exception as e:
        logger.warning("health_check_failed", service="postgresql", error=str(e))
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }


async def _check_redis() -> dict[str, Any]:
    """Test Redis connectivity."""
    start = time.time()
    try:
        from redis.asyncio import Redis

        client = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            db=settings.REDIS_DB,
            socket_timeout=5,
        )
        pong = await client.ping()
        info = await client.info("server")
        await client.aclose()
        return {
            "status": "healthy" if pong else "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "version": info.get("redis_version", "unknown"),
        }
    except Exception as e:
        logger.warning("health_check_failed", service="redis", error=str(e))
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }


async def _check_qdrant() -> dict[str, Any]:
    """Test Qdrant connectivity."""
    start = time.time()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/healthz"
            resp = await client.get(url)
            return {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "latency_ms": round((time.time() - start) * 1000, 2),
            }
    except Exception as e:
        logger.warning("health_check_failed", service="qdrant", error=str(e))
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }


async def _check_elasticsearch() -> dict[str, Any]:
    """Test Elasticsearch connectivity."""
    start = time.time()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            url = f"http://{settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}"
            resp = await client.get(url)
            data = resp.json()
            return {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "latency_ms": round((time.time() - start) * 1000, 2),
                "version": data.get("version", {}).get("number", "unknown"),
                "cluster": data.get("cluster_name", "unknown"),
            }
    except Exception as e:
        logger.warning("health_check_failed", service="elasticsearch", error=str(e))
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }


async def _check_neo4j() -> dict[str, Any]:
    """Test Neo4j connectivity."""
    start = time.time()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            url = f"http://{settings.NEO4J_HOST}:{settings.NEO4J_HTTP_PORT}"
            resp = await client.get(url)
            return {
                "status": "healthy" if resp.status_code == 200 else "unhealthy",
                "latency_ms": round((time.time() - start) * 1000, 2),
            }
    except Exception as e:
        logger.warning("health_check_failed", service="neo4j", error=str(e))
        return {
            "status": "unhealthy",
            "latency_ms": round((time.time() - start) * 1000, 2),
            "error": str(e),
        }