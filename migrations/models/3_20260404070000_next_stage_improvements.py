from tortoise import BaseDBAsyncClient

RUN_IN_TRANSACTION = True


def _sqlite_upgrade_sql() -> str:
    return """
        ALTER TABLE "findings" ADD COLUMN "root_cause_class" VARCHAR(64);
        ALTER TABLE "patches" ADD COLUMN "metadata" JSON;
        UPDATE "patches" SET "metadata" = '{}' WHERE "metadata" IS NULL;
        ALTER TABLE "agent_handoffs" ADD COLUMN "kind" VARCHAR(32) NOT NULL DEFAULT 'code';
        ALTER TABLE "agent_handoffs" ADD COLUMN "confidence" REAL NOT NULL DEFAULT 0.5;
        ALTER TABLE "skill_candidate_records" ADD COLUMN "cooldown_until" TIMESTAMPTZ;
        ALTER TABLE "skill_evaluation_records" ADD COLUMN "mode" VARCHAR(32) NOT NULL DEFAULT 'heuristic';"""


def _postgres_upgrade_sql() -> str:
    return """
        ALTER TABLE "findings" ADD COLUMN IF NOT EXISTS "root_cause_class" VARCHAR(64);
        ALTER TABLE "patches" ADD COLUMN IF NOT EXISTS "metadata" JSONB NOT NULL DEFAULT '{}'::jsonb;
        ALTER TABLE "agent_handoffs" ADD COLUMN IF NOT EXISTS "kind" VARCHAR(32) NOT NULL DEFAULT 'code';
        ALTER TABLE "agent_handoffs" ADD COLUMN IF NOT EXISTS "confidence" DOUBLE PRECISION NOT NULL DEFAULT 0.5;
        ALTER TABLE "skill_candidate_records" ADD COLUMN IF NOT EXISTS "cooldown_until" TIMESTAMPTZ;
        ALTER TABLE "skill_evaluation_records" ADD COLUMN IF NOT EXISTS "mode" VARCHAR(32) NOT NULL DEFAULT 'heuristic';"""


async def upgrade(db: BaseDBAsyncClient) -> str:
    if db.capabilities.dialect == "sqlite":
        return _sqlite_upgrade_sql()
    return _postgres_upgrade_sql()


async def downgrade(db: BaseDBAsyncClient) -> str:
    if db.capabilities.dialect == "sqlite":
        return ""
    return """
        ALTER TABLE "skill_evaluation_records" DROP COLUMN IF EXISTS "mode";
        ALTER TABLE "skill_candidate_records" DROP COLUMN IF EXISTS "cooldown_until";
        ALTER TABLE "agent_handoffs" DROP COLUMN IF EXISTS "confidence";
        ALTER TABLE "agent_handoffs" DROP COLUMN IF EXISTS "kind";
        ALTER TABLE "patches" DROP COLUMN IF EXISTS "metadata";
        ALTER TABLE "findings" DROP COLUMN IF EXISTS "root_cause_class";"""


MODELS_STATE = "eJyrVsrNT0nNKVayUqhWKs7OzMmJL0hMzo4vSk3OL0qBCJckJuWkAlnY5GtrawEt2hhv"
