# Verified R7 Resume Locations

23 November addition: both original trained adapters are now privately on HF.
R6 commit: `bb5d0466e753d0f02c6633042172abe48339438a`; its separate nine-file
download/hash check passed. R7 commit: `bc5a25dc6772a7503827936c491647a575279495`;
remote hashes and the separate nine-file byte-restoration check passed.
R7's HF package is explicitly inference-only. Full training recovery is retained
locally and on Modal; uploading it remains pending after repeated timeouts.
Read [R6_R7_HF_BACKUPS.md](R6_R7_HF_BACKUPS.md) for exact package identities,
upload/verification steps and recovery boundaries. Keep the existing copies below.
The approved cloud app shutdown changed no corpus or model volume. The additional
20-cache cleanup is completed and verified; no further deletion is authorized.

Verified 18 November 2026 UTC. Modal volume: `ghana-health-understand-train`.

Full data/code snapshot (607,656,162 bytes, SHA256 is the filename):
`/research-handoffs/twi-understanding/2026-11-17-r7/673ac4802412935319fe6871f8dd06967511c5a79f747a8fa5f78b46f663f0a5.tar.gz`

Final Hermes supplement, 21 files plus a hash manifest, all read back and verified:
`/research-handoffs/twi-understanding/2026-11-17-r7/hermes/56ef236e7e19febea68c8bc00510fdc29c0748a9aa6aa1e3a488badf61e47d8f/`

Read `MANIFEST.json` in that directory. It contains the exact training receipt,
completed smoke evidence, completed R6 tool evaluation, latest execution map and
`docs/research/R7_HERMES_HANDOFF.md`. The supplement binds commit `aa60fc4`.
Use it with the full snapshot, not an older backup alone. Preserve file paths.

Training call `fc-01M7P3443CKHMZGBK47M0V2CCG` subsequently completed. The remote
archives above are historical snapshots; do not interpret their submitted state
as permission to relaunch. Local completion evidence, both adapters and final R7
recovery state are verified in the private off-provider archive:

`/Users/welcome/Documents/Research-Exports/ghana-health-ai/r7-offline-recovery-20261118.tar.gz`

SHA256: `7fc173660b9b82cd77efa7e8568c212e7bc97580f260b7e040c5ec72b3ef5fd2`.
This archive excludes the entire research corpus and public base weights.
The completed cache cleanup did not modify the remote research backup volume.
The latest restrictions and next steps are in [the handoff](R7_HERMES_HANDOFF.md)
and [local testing instructions](R7_LOCAL_MAC.md). No R7 quality gain is claimed.
