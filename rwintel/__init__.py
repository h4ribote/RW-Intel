"""RW-Intel: the control side of a machine learning player for Rusted Warfare.

Three parts, matching the documents under `docs/system`:

- `data` reads what the game ships on disk, without launching it: maps and unit definitions.
- `wire` is the frame format spoken between the in-process agent and this side.
- `control` is the process the agents connect to; it holds the policies and drives the episodes.
"""
