# Third-Party Notices

This repository contains **original wrapper code** to run CAT12 preprocessing and longitudinal statistics in a reproducible way.

## Scope of this notice

- The license in `LICENSE` applies to the **original code and documentation in this repository** (the “wrapper”).
- **CAT12, SPM12, MATLAB Runtime, and other third-party tools are NOT included in this repository**.
- The installation script downloads third-party software from upstream sources. Your use of those components is governed by **their own licenses/terms**.

## Third-party software downloaded/used by this project

This project is **not affiliated with or endorsed by** the CAT12 or SPM developers, nor by The MathWorks.

### CAT12 (includes SPM12 standalone integration)
- Upstream: https://github.com/ChristianGaser/cat12
- This project downloads CAT12 standalone from upstream releases.
- CAT12/SPM12 remain copyright of their respective authors.

### MATLAB Runtime (MCR) R2017b (v93)
- Upstream (MathWorks): https://www.mathworks.com/products/compiler/matlab-runtime.html
- This project downloads the MATLAB Runtime installer from MathWorks sources.
- MATLAB is a registered trademark of The MathWorks, Inc.

### Deno (used for BIDS validation tooling)
- Upstream: https://deno.com/
- Installed into the repository-local `external/` directory by the installer.

### bids-validator (JavaScript BIDS Validator)
- Upstream: https://github.com/bids-standard/bids-validator
- This project invokes the validator via the installed tooling.

## Academic / research intent

This wrapper is intended for academic/research workflows. That said, third-party licenses may still impose requirements regardless of intent; please review and comply with the upstream terms for all downloaded components.
