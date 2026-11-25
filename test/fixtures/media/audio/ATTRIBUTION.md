# Fixture attribution

The `*.win.f32.gz` files are raw audio from the **FLEURS** dataset (`google/fleurs`, configuration `pt_br`, split `dev`),
released under **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>; license field of the dataset card on Hugging Face: `cc-by-4.0`).

FLEURS: Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera,
Ankur Bapna. *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*. 2022. arXiv:2205.12446.

| window | made from (FLEURS id, audio file) | sentence (dataset `raw` transcript) |
|---|---|---|
| `A0` | 1578, `10165966771567591302.wav` | Então, Lakkha Singh tomou a frente para cantar os bhajans. |
| `B0` | the last 8 s of two clips played one after the other: 1621, `10221738448089878097.wav` (its final 2.36 s), then 1552, `1088555081702852536.wav` (whole) | No começo, as vestimentas foram muito influenciadas pela cultura bizantina na região leste. / O resultado da análise do gráfico será disponibilizado no site público. |

Changes: the dataset's 16 kHz mono float WAV samples were kept as float32 and cut or joined into one 8 s window (A0: the clip, right-aligned
and padded with zeros; B0: the tail of one clip joined to the next), then stored gzip-compressed as raw little-endian float32.
Nothing was filtered or re-recorded. The other files (`*.mel_torch`, `*.h9`, `*.h12`, `*.h18`) are numbers computed from these windows by the
torch Marimba package and by the JavaScript log-mel and the ONNX encoder; they are derived from the same audio and carry the same attribution.
