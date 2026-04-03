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
