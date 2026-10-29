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
