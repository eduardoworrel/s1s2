# Recipes

Records of how the media packages were made from third-party checkpoints (audio: Marimba on Qwen3-ASR-0.6B). `SOURCES.sha256` lists the sha256 of every file copied from the reference runs and of the weight files that
`manifests/*.json` pin. `make-manifests.mjs` writes the manifest from a weights directory.

Nothing in here is run by `npm test`, and no weight file is in the repository (`.weights/` is ignored).
