---
name: task-board-sync
description: Synchronize the Tunix CrafText roadmap, thematic Kanban, dashboard and documentation after any task, status, dependency, implementation or planning change. Use after completing a change, changing task text/status/dependencies, preparing a commit, or when the board may have drifted from documentation.
---

# Task board sync

Run `python .codex/skills/task-board-sync/scripts/sync_task_views.py` from the repository root
after modifying tasks or implementation status. It is the required final step before the normal
audit/commit/site-build sequence.

## Workflow

1. Keep task status in `docs/plan.md`: `[x]` done, `[~]` active, `[ ]` planned. Do not copy cards
   into generated Markdown.
2. If `docs/tasks.json` exists, make it the task/dependency source of truth and let the script
   validate IDs, unknown dependencies and dependency cycles before publishing a status.
3. Run the sync script. It rebuilds the dashboard/Kanban/site and verifies every roadmap task
   produced one Kanban card.
4. Review the generated Kanban: a completed task must have completed dependencies; an active task
   must accurately name the current implementation.
5. Update relevant architecture/quality docs and `docs/project_status.json`, then run `make verify`
   and commit only the source files. Generated pages remain ignored build output.

## Failure policy

Treat unknown task IDs, dependency cycles, missing cards and failed site builds as blockers. Do not
manually patch `docs/_generated/`; repair the source task data or roadmap and re-run sync.
