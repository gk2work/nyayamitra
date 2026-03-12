"""
NyayaMitra — Database Connection and Session Management.

Provides:
- Async SQLAlchemy engine and session factory
- Database initialization (create all tables)
- Dependency injection for FastAPI endpoints

Usage:
    # In FastAPI endpoints:
    from app.database import get_db
    async def my_endpoint(db: AsyncSession = Depends(get_db)):
        ...

    # To initialize tables:
    from app.database import init_db
    await init_db()
"""

from collections.abc import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

logger = structlog.get_logger()

# ─── Async Engine ────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.database_url,
    echo=settings.APP_DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# ─── Session Factory ─────────────────────────────────────────────────────────
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─── Dependency Injection ────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a database session for FastAPI dependency injection.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ─── Table Initialization ────────────────────────────────────────────────────
async def init_db() -> None:
    """
    Create all database tables if they don't exist.

    Call this during application startup.
    For production, use Alembic migrations instead.
    """
    from app.models.legal import Base

    logger.info("database_init_start", url=settings.POSTGRES_HOST)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("database_init_complete", tables=list(Base.metadata.tables.keys()))


async def drop_db() -> None:
    """
    Drop all database tables. USE WITH CAUTION.

    Only for development/testing. Never call in production.
    """
    from app.models.legal import Base

    logger.warning("database_drop_start")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.warning("database_drop_complete")