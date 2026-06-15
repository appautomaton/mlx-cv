# Host Tools

Host: `opencode`

Use this file when an Automaton skill asks for host-native collaboration or coordination tools.

## Automaton Subagents

`installHost()` wrote these host-native subagents into this host's agent directory:

- `automaton-implementer`: Implements exactly one approved Automaton plan slice from coordinator-provided context and returns evidence. (execute stage; dispatched by auto-execute)
- `automaton-spec-reviewer`: Reviews spec compliance for one approved Automaton plan slice. Verdict only; no edits. (execute stage; dispatched by auto-execute)
- `automaton-quality-reviewer`: Reviews maintainability and regression risk for one approved Automaton plan slice. Verdict only; no edits. (execute stage; dispatched by auto-execute)
- `automaton-librarian`: Read-only codebase explorer. Answers where/how/which-files questions and returns a bounded, anchored map. Evidence only; no edits, no decisions. (any stage; read-only one-shot lookup)

Their static role bodies are baked into the host agent files. Execute-stage agents take per-call slots from `auto-execute/references/*-prompt.md` (slice, constraints, acceptance criteria, implementation summary). The read-only `automaton-librarian` is governed by `.agent/.automaton/references/LIBRARIAN.md` and may be dispatched from any stage.

## Dispatch

- availability: available
- dispatch: Use the Task tool (or `@mention` where supported) to invoke the named automaton agent you are dispatching — `automaton-implementer`, `automaton-spec-reviewer`, `automaton-quality-reviewer`, or `automaton-librarian` (see the roster above). For the execute-stage agents pass the per-call dispatch packet (slice, constraints, acceptance criteria, implementation summary) from `auto-execute/references/*-prompt.md` as the task body; for the read-only `automaton-librarian` pass the bounded question packet from `.agent/.automaton/references/LIBRARIAN.md`. The role body is in the markdown file under `.opencode/agents/` and every Automaton subagent denies `permission.task` so it cannot fan out to another subagent.
- wait: Wait for the OpenCode subagent response before dispatching dependent reviews.
- cleanup: No Automaton cleanup step is required; follow OpenCode session conventions.
- tracking: Use todowrite for session-local progress tracking when useful.
- isolation: No native worktree parameter: for plan-approved parallel cross-slice dispatch the coordinator runs `git worktree add` first and scopes each task to its worktree directory. Integrate and remove per `auto-execute/references/git-rhythm.md` (Parallel Isolation).
- precondition: The primary agent's `permission.task` configuration must allow `automaton-implementer`, `automaton-spec-reviewer`, `automaton-quality-reviewer`, and `automaton-librarian` for Task-tool named-agent dispatch to work. If any of those four names is denied or filtered out, treat dispatch of that agent as unavailable and stop under SUBAGENT-PROTOCOL.md's "Host does not expose subagent support" condition rather than pasting a role body into a generic agent.

## Rules

- Follow the skill protocol first; this file only maps host tool names.
- Dispatch only by named agent (`automaton-implementer`, `automaton-spec-reviewer`, `automaton-quality-reviewer`, `automaton-librarian`). Do not paste a role body into a generic worker, explorer, or other host agent at runtime.
- If the host cannot expose one of the named agents (configuration disabled, permission denied, capability missing), stop under SUBAGENT-PROTOCOL.md's "Host does not expose subagent support" condition. Do not fall back to runtime-curated prompt injection.
- Do not invent a universal SDK or CLI when the host has native subagent tools.
