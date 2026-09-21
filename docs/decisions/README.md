# Decision records

Short, dated notes on choices that are expensive to revisit. One file per decision,
numbered, never edited after the fact — a later decision supersedes an earlier one rather
than rewriting it.

The eight decisions taken in Milestone 1 are recorded together in
`docs/IMPLEMENTATION_PLAN.md` (D1–D8), because they were made as one set while designing
the architecture. From Milestone 7 onward each decision gets its own file.

## Template

```markdown
# NNNN — <decision>

Date: YYYY-MM-DD
Status: proposed | accepted | superseded by NNNN

## Context
What question came up, and what constrained the answer.

## Evidence
The numbers, with a link to the run directory that produced them.

## Decision
What was chosen.

## Licence check
For an engine decision: the weight licence, whether commercial use is permitted, and who
confirmed it.

## Consequences
What this makes easy, what it makes hard, and what would reverse it.
```

## Expected records

| # | Decision | Milestone |
|---|---|---|
| 0001 | Primary character voice engine | 7 |
| 0002 | Whether optional cleaning stages helped this pack | 2 / 6 |
| 0003 | Streaming strategy and chunk boundaries | 12 |
| 0004 | Barge-in policy: when to cancel the LLM stream | 13 |
