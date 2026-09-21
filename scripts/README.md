# Scripts

Thin wrappers over library code — logic lives in `services/`, never here, so that
anything a script does can also be done from a test or the API.

- `preprocessing/` — Voice Pack pipeline stages (Milestone 2)
- `training/` — per-engine dataset export and training launchers (Milestones 3-5)
- `evaluation/` — benchmark helpers (Milestone 6)

The two main entry points are installed as commands:

    cvai-voicepack   init | validate | list
    cvai-bench       run | blind | aggregate | report | demo
