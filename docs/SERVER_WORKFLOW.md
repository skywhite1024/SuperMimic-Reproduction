# Local and server Git workflow

The GitHub repository is the canonical public source. Each machine checks out an explicit commit or release tag. Keep runtime data, checkpoints and trajectories outside tracked source paths.

```bash
git clone https://github.com/skywhite1024/SuperMimic-Reproduction.git
cd SuperMimic-Reproduction
git checkout --detach v0.1.0
python tools/check_release.py
```

Before adopting the public repository on an existing research server, preserve an allowlisted source snapshot in a separate **private local Git history**. Compare its file hashes against `SOURCE_PROVENANCE.json`'s original hashes. Do not overwrite old experiment directories or push account-specific startup commands, keys, environment directories, raw datasets or model weights into public Git history.

For a new experiment, create a branch, commit the code/configuration first, and record `git rev-parse HEAD`, dirty status, configuration, data hashes and parent-checkpoint hashes alongside the outputs. Commit deliberate source changes on the server to that same repository and review the diff before pushing. Preserve every evaluation failure and the original complete denominator.

SenseCore-specific operators must check current institutional location/network requirements, account-wide activity and quota before requesting resources. A source audit needs no GPU. Use a finite CPU program, preserve its log and receipt, and verify the platform terminal state. A process exiting is not sufficient evidence that a persistent interactive application was stopped.

## Verified adoption on 2026-10-29

Local publication and the server canonical checkout were both verified at `623d190800e5204a3d430998ccbc4980173d5e1e` (`v0.1.0`). The server checkout passed `tools/check_release.py` and had a clean worktree. The separate private AFS audit history ended at `005ea6e241069c413930f1b8f1e57f5c7c9907a6`; it keeps original historical bytes and the comparison receipt.

The audit found 39 exact source matches, one identical file under another name, one one-line checkpoint-path difference, and eight additional preparation/postprocessing scripts. Those eight are now included at their historical project-relative paths. The differing training file is preserved under `server-history/20261029/`. A bootstrap environment's 495 pip source files are excluded from the public repository. Historical cloud shell scripts expect the old staged AFS layout and `USER_ID` substitution; they are archival examples, not the default installation path.

AFS file ownership differs from the container's effective UID. Git commands in the audit used `-c safe.directory=<exact personal repository path>` for the two explicitly created repositories only. No global wildcard trust, SSH setting, credential or existing experiment file was changed. One initial failed Git configuration attempt was preserved, then one corrected snapshot job and one short source-reconciliation job succeeded. All requested zero GPUs; no training was run.

See `validation/cloud-source-reconciliation-20261029.json` for counts and hashes. Later commits on `main` add the recovered archive and publication receipts; the tested release snapshot and server checkout stay pinned to the immutable v0.1.0 tag.
