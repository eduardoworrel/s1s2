import { build } from "esbuild";
const common = { format: "esm", target: "es2022", bundle: true, packages: "external", logLevel: "warning" };
await build({ ...common, entryPoints: ["src/index.ts"], outfile: "dist/index.js" });
await build({ ...common, entryPoints: ["src/fidelity.ts"], outfile: "dist/fidelity.js" });
await build({ ...common, entryPoints: ["src/media/index.ts"], outfile: "dist/media.js" });
await build({ ...common, entryPoints: ["src/manifest.ts"], outfile: "dist/manifest.js" });
await build({ ...common, entryPoints: ["src/package.ts"], outfile: "dist/package.js" });
