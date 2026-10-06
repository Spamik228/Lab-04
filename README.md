[![CI](https://github.com/Spamik228/Lab-04/actions/workflows/ci.yml/badge.svg)](https://github.com/Spamik228/Lab-04/actions/workflows/ci.yml)
## Installation

### From local source (Development)
To install `findex` directly into your global environment PATH from local source:

```bash
uv tool install .

From Wheel (Release)
To build the distribution packages and install findex from the built wheel file:

Bash
# 1. Build wheel and source distributions
uv build

# 2. Install CLI executable from built wheel
uv tool install dist/findex4-0.4.0-py3-none-any.whl.0-py3-none-any.whl

From GitHub Release
To install from a clean environment using a direct URL to the published wheel asset:
uv tool install [https://github.com/Spamik228/Lab-04/releases/download/v0.4.0/findex4-0.4.0-py3-none-any.whl](https://github.com/Spamik228/Lab-04/releases/download/v0.4.0/findex4-0.4.0-py3-none-any.whl)

Verification
Verify that findex is available on your PATH:

Bash
findex --help