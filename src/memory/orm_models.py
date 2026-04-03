from __future__ import annotations

from tortoise import fields
from tortoise.models import Model


class RunRecord(Model):
    run_id = fields.CharField(max_length=32, primary_key=True)
    workflow_name = fields.CharField(max_length=128)
    repo_root = fields.TextField()
    started_at = fields.DatetimeField()
    finished_at = fields.DatetimeField(null=True)
    status = fields.CharField(max_length=32)
    summary = fields.TextField(null=True)
    report_dir = fields.TextField(null=True)

    class Meta:
        table = "runs"


class SnapshotRecord(Model):
    run_id = fields.CharField(max_length=32, primary_key=True)
    repo_root = fields.TextField()
    scanned_at = fields.DatetimeField()
    revision = fields.TextField(null=True)
    file_hashes = fields.JSONField()

    class Meta:
        table = "snapshots"


class TaskRecord(Model):
    task_id = fields.CharField(max_length=64, primary_key=True)
    run_id = fields.CharField(max_length=32, db_index=True)
    agent_kind = fields.CharField(max_length=32)
    task_type = fields.CharField(max_length=32)
    targets = fields.JSONField()
    payload_summary = fields.JSONField()
    created_at = fields.DatetimeField()
    status = fields.CharField(max_length=32)
    summary = fields.TextField()

    class Meta:
        table = "tasks"


class FindingRecord(Model):
    id = fields.IntField(primary_key=True)
    run_id = fields.CharField(max_length=32, db_index=True)
    task_id = fields.CharField(max_length=64, db_index=True)
    agent_kind = fields.CharField(max_length=32)
    severity = fields.CharField(max_length=32)
    rule_id = fields.CharField(max_length=255)
    category = fields.CharField(max_length=64)
    path = fields.TextField(null=True)
    line = fields.IntField(null=True)
    symbol = fields.TextField(null=True)
    message = fields.TextField()
    fingerprint = fields.TextField()
    evidence = fields.JSONField()
    state = fields.CharField(max_length=32)

    class Meta:
        table = "findings"


class PatchRecord(Model):
    run_id = fields.CharField(max_length=32, primary_key=True)
    summary = fields.TextField()
    rationale = fields.TextField()
    touched_files = fields.JSONField()
    diff_text = fields.TextField()
    suggestions = fields.JSONField()
    applied = fields.BooleanField()
    file_patches = fields.JSONField()

    class Meta:
        table = "patches"


class IssueCatalogRecord(Model):
    id = fields.IntField(primary_key=True)
    repo_root = fields.TextField()
    fingerprint = fields.TextField()
    rule_id = fields.CharField(max_length=255)
    path = fields.TextField(null=True)
    message = fields.TextField()
    status = fields.CharField(max_length=32)
    first_seen_run = fields.CharField(max_length=32)
    last_seen_run = fields.CharField(max_length=32)
    occurrences = fields.IntField()

    class Meta:
        table = "issue_catalog"
        unique_together = (("repo_root", "fingerprint"),)


class AgentSessionRecord(Model):
    session_id = fields.CharField(max_length=32, primary_key=True)
    run_id = fields.CharField(max_length=32, db_index=True)
    task_id = fields.CharField(max_length=64, db_index=True)
    agent_kind = fields.CharField(max_length=32)
    task_type = fields.CharField(max_length=32)
    working_repo_root = fields.TextField()
    objective = fields.TextField()
    started_at = fields.DatetimeField()
    finished_at = fields.DatetimeField(null=True)
    completion_reason = fields.CharField(max_length=64, null=True)
    summary = fields.TextField(null=True)
    step_count = fields.IntField(default=0)
    tool_call_count = fields.IntField(default=0)
    active_skill_version = fields.CharField(max_length=128, null=True)
    candidate_skill_version = fields.CharField(max_length=128, null=True)
    skill_profile_hash = fields.CharField(max_length=128, null=True)

    class Meta:
        table = "agent_sessions"


class AgentStepRecord(Model):
    id = fields.IntField(primary_key=True)
    session_id = fields.CharField(max_length=32, db_index=True)
    step_index = fields.IntField()
    decision_summary = fields.TextField()
    action_type = fields.CharField(max_length=32)
    tool_name = fields.CharField(max_length=128, null=True)
    tool_input = fields.JSONField()
    final_response = fields.JSONField()
    created_at = fields.DatetimeField()

    class Meta:
        table = "agent_steps"


class AgentHandoffRecord(Model):
    id = fields.IntField(primary_key=True)
    session_id = fields.CharField(max_length=32, db_index=True)
    run_id = fields.CharField(max_length=32, db_index=True)
    task_id = fields.CharField(max_length=64, db_index=True)
    source_agent = fields.CharField(max_length=32)
    title = fields.TextField()
    description = fields.TextField()
    recommended_change = fields.TextField()
    severity = fields.CharField(max_length=32)
    affected_files = fields.JSONField()
    evidence = fields.JSONField()
    metadata = fields.JSONField()

    class Meta:
        table = "agent_handoffs"


class ToolCallRecord(Model):
    id = fields.IntField(primary_key=True)
    session_id = fields.CharField(max_length=32, db_index=True)
    step_index = fields.IntField()
    tool_name = fields.CharField(max_length=128)
    status = fields.CharField(max_length=32)
    tool_input = fields.JSONField()
    output = fields.JSONField()
    summary = fields.TextField(null=True)
    error = fields.TextField(null=True)
    active_skill_version = fields.CharField(max_length=128, null=True)
    candidate_skill_version = fields.CharField(max_length=128, null=True)
    skill_profile_hash = fields.CharField(max_length=128, null=True)
    started_at = fields.DatetimeField()
    finished_at = fields.DatetimeField()

    class Meta:
        table = "tool_calls"


class SkillPackRecord(Model):
    id = fields.IntField(primary_key=True)
    repo_root = fields.TextField()
    agent_kind = fields.CharField(max_length=32)
    name = fields.CharField(max_length=128)
    version = fields.CharField(max_length=128)
    description = fields.TextField()
    status = fields.CharField(max_length=32)
    source = fields.CharField(max_length=32)
    system_prompt = fields.TextField()
    skill_markdown = fields.TextField()
    examples = fields.JSONField()
    policy = fields.JSONField()
    profile_hash = fields.CharField(max_length=128)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "skill_pack_records"
        unique_together = (("repo_root", "agent_kind", "version"),)


class SkillBindingRecord(Model):
    id = fields.IntField(primary_key=True)
    repo_root = fields.TextField()
    agent_kind = fields.CharField(max_length=32)
    active_version = fields.CharField(max_length=128)
    source = fields.CharField(max_length=32)
    frozen = fields.BooleanField(default=False)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "skill_binding_records"
        unique_together = (("repo_root", "agent_kind"),)


class SkillCandidateRecord(Model):
    candidate_id = fields.CharField(max_length=32, primary_key=True)
    repo_root = fields.TextField()
    agent_kind = fields.CharField(max_length=32)
    based_on_version = fields.CharField(max_length=128)
    version = fields.CharField(max_length=128)
    status = fields.CharField(max_length=32)
    shadow_runs = fields.IntField(default=0)
    notes = fields.JSONField()
    skill_payload = fields.JSONField()
    created_at = fields.DatetimeField()
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "skill_candidate_records"


class SkillEvaluationRecord(Model):
    evaluation_id = fields.CharField(max_length=32, primary_key=True)
    repo_root = fields.TextField()
    agent_kind = fields.CharField(max_length=32)
    run_id = fields.CharField(max_length=32, db_index=True)
    active_version = fields.CharField(max_length=128)
    candidate_version = fields.CharField(max_length=128, null=True)
    active_score = fields.FloatField()
    candidate_score = fields.FloatField(null=True)
    promoted = fields.BooleanField(default=False)
    reasons = fields.JSONField()
    created_at = fields.DatetimeField()

    class Meta:
        table = "skill_evaluation_records"


class AgentReflectionRecord(Model):
    reflection_id = fields.CharField(max_length=32, primary_key=True)
    repo_root = fields.TextField()
    run_id = fields.CharField(max_length=32, db_index=True)
    task_id = fields.CharField(max_length=64, db_index=True)
    session_id = fields.CharField(max_length=32, db_index=True)
    agent_kind = fields.CharField(max_length=32)
    skill_version = fields.CharField(max_length=128)
    summary = fields.TextField()
    metrics = fields.JSONField()
    upgrade_hints = fields.JSONField()
    created_at = fields.DatetimeField()

    class Meta:
        table = "agent_reflection_records"
