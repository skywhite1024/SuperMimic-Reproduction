# Qualified-expert V2 source preparation

Status: V2 has not been trained or adopted. The full experiment goal remains incomplete. This directory adds a reviewed source archive, a disabled public startup template and verification metadata. Existing published v0.1.0 and v0.2.0 results are unchanged.

The proposed finite job uses one H100, 4 CPU cores and 64 GiB RAM. One GPU remains allocated while the CPU preflight hides CUDA. Initialization includes downloading, unpacking and validation within the same 600-second deadline (585 seconds of work plus cleanup); the whole job is limited to 5,400 seconds. The strict raw and calibrated teacher checks must pass before any of at most 1,000 updates, including 200 world-model-only updates. A teacher-gate failure requires zero updates. No automatic retry or extra training block is permitted.

The unchanged goal is one deployed policy for all 55 clips (15 new plus 40 old), tracking and contact quality, and three 120-second continuous trials per evaluation mode with zero resets after the initial state. Preparation and process completion do not establish this acceptance.

The archive is 76,418 bytes. Its SHA256 is `f7afc8dbe99bc5711a995b021c62dc862e1b2e17004c4d635fd81d770a3833dd`. Pin the raw download URL to the containing 40-character Git commit. The public startup template retains placeholders and does not submit a job. Private commands, models, raw motion data and runtime logs are excluded.

Offline tests: 27 passed and one native Linux group test skipped on Windows. Windows leaf-process completion and mocked Linux signal logic do not prove native Linux behavior. The job must pass its native process-group timeout/KILL self-test before the GPU probe, CPU preflight or scientific execution. The original standalone CPU2/RAM8 guard and all scientific files, frozen plan, canonical members, physics, thresholds and evaluation denominators are preserved. Current account, network, resource and duplicate checks remain required before cloud execution. BONES remains paused.
