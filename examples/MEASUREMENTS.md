# Text demos: measurements (24 Nov 2026)

Exploratory: tuning data, no locked test set. One Mac (M2 Pro, 16 GB, on battery), Chrome 158 headless with WebGPU, onnxruntime-web 1.31.1, S5-nano int4 with the ContractNLI / CLINC150 / SIB-200 heads, model file hashes equal to `manifest.json`, served from a local mirror. One demo per page, one cold load at a time.

| demo | items | decisions | margin (per item) | ms per decision | cold load |
|---|---|---|---|---|---|
| topic router | 8 sentences | 8/8 routed, 0 abstained; 7/7 sentences with a clear topic correct; "Please reset my password." (off topic) was routed to science/technology with margin 1.00 | 1.00 on all | 792 (page mean; 696 to 892) | 17.6 s |
| contract (NDA, 8 claims) | 8 | 8 answered; the first (sharing with third parties) = No (the text says the opposite), correct | 0.79; 1.00; 0.95; 0.98; 0.95; 1.00; 0.36; 1.00 | 919 per claim (430-token contract) | 17.4 s |
| sentinel (28 messages) | 28 | 11 woke S2; 0/11 urgent messages missed; 0/17 non-urgent woken | 0.66 to 1.00 (urgent), 0.96 to 1.00 (routine) | S1 median 506; S2 per summary median 1343 | 17.0 s |
| replay of 300 messages (sentinel by logits) | 300 (36 urgent) | AUROC 1.000; threshold 0.5: 33/36 urgent woken (3 missed), 0/264 false; threshold 0.1: 0 missed, 0 false; threshold 0.05: 0 missed, 1 false | median 0.999 | median 667 (p90 838); 300 decisions in 209 s | 17.5 s |

The margin does not separate out-of-scope input: the off-topic sentence got margin 1.00.

Decide vs generate JSON (router, first sentence, median of 3, this machine): deciding 820 ms against 686 ms for generating the JSON (ratio 0.8). With S5-nano here, deciding by head was not cheaper than generating the JSON of one sentence. The contract demo showed the quoted passage, checked against the text, on the 3 claims where it was requested. No horizontal overflow in any of the 8 captures with a result (desktop 1280 and phone 390), nor in the 4 without.

The sentinel S1 time (506 ms) and the replay time (667 ms) measure the same thing under different conditions (28 messages streaming against 300 in a probe loop); the difference was not investigated.

## policy-checker with onnxruntime-web 1.31.1

- As the page is written, with a local mirror of the model: **it does not create the session**. It fails with "large file not on the Hugging Face Hub (or size not pinned) ... pass allowUnverified: true". The other three demos pass `allowUnverified` when a mirror is used; `examples/policy-checker/app.js` does not.
- With `allowUnverified: true` injected only in the test server's response (the file in the repository was not changed) and onnxruntime-web 1.31.1 served from the repository root: the session is created with the S5-nano package (not the GDPR/LGPD model named in `requirements.json`, which is on Hugging Face and was not read), "Model ready", button enabled, 15 s, `ort.env.versions.web` = 1.31.1, no request outside localhost. No policy check was run (it needs the GDPR model).
- `examples/policy-checker/serve.mjs` serves onnxruntime-web from `node_modules/@huggingface/transformers/node_modules`, which is **1.31.0-dev.20260914**, not the 1.31.1 in `package.json`. Read, not run.

## Limits

Small n in the demos (8, 8, 28); only the 300-message replay has enough positives for AUROC (36 urgent). AUROC 1.000 with 3 urgent messages under 0.2 shows a perfect ranking with a poor threshold, not calibration. Mac on battery; one cold load per demo. No comparison with bf16. The tokenizer of the mirror is not the manifest file; the sha256 check passed, but the origin of the file used (a browser profile cache) was not confirmed. Weights downloaded directly from Hugging Face were not tested.
