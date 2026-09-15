# unichat
Universal API chat Python client for OpenAI, MistralAI, Anthropic, xAI, Google AI, or any OpenAI SDK LLM provider.

## Clean installation from source

Prerequisites: Git, Python 3.11 or newer (as declared in `pyproject.toml`), and `uv` available on your PATH. The shell commands below use a POSIX shell.

Enter a fresh checkout, create an isolated environment, and install the package and its declared dependencies:

```shell
git clone https://github.com/amidabuddha/unichat.git
cd unichat
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e .
```

Ensure `python3` selects Python 3.11 or newer. There is no dependency lockfile in this repository, so installation resolves the version ranges in `pyproject.toml`; dependencies are not frozen. There are no declared development dependency groups. Keep dependencies in `.venv`, not in a global or Homebrew Python installation.

Build the source archive and wheel, then run the interactive sample:

```shell
uv build --out-dir dist
.venv/bin/python sample_chat.py
```

`uv build` uses the declared setuptools backend and an isolated build environment; it does not require a globally installed `build` frontend. Packaging is not required to run the editable installation.

The sample prompts for your provider API key, model name, streaming preference, and system instructions. It makes live API calls. No environment-variable or `.env` setup is required by the sample; enter your own API key at the prompt and keep it out of source control.

## Normal development

Run these commands from the repository root. The editable installation uses your working source, so ordinary Python edits only require restarting the sample:

```shell
.venv/bin/python sample_chat.py
```

There is no configured watch command or automated test suite; `sample_chat.py` is the existing manual functionality check. Ordinary source changes do not require deleting dependencies or a full rebuild. If dependency declarations or package metadata change, repeat the editable installation:

```shell
uv pip install --python .venv/bin/python -e .
```

When you need updated distributable archives, rebuild without cleaning:

```shell
uv build --out-dir dist
```

## Clean rebuild of an existing checkout

From the repository root, remove only the project-local environment and generated packaging output, then restore dependencies and build:

```shell
rm -rf .venv dist build unichat.egg-info
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -e .
uv build --out-dir dist
```

Here `.venv` is the disposable environment created above, `dist` holds package archives, `build` is setuptools build output, and `unichat.egg-info` is generated package metadata. These paths must contain only their generated contents. This sequence preserves source, configuration, `.env` files outside `.venv`, user data outside these generated directories, and any lockfiles. It does not clear global caches or toolchains. Because this repository has no lockfile, restored dependency versions may differ from the previous installation.

To check functionality afterward, run `.venv/bin/python sample_chat.py` again.

## Usage:

1. To use the published package instead of a source checkout, install it in an isolated environment in your application directory:

```shell
uv venv --python python3 .venv
uv pip install --python .venv/bin/python unichat
```

2. Import `UnifiedChatApi` in your application. Optionally import `MODELS_LIST` for additional validation:

```python
from unichat import UnifiedChatApi, MODELS_LIST

client = UnifiedChatApi(api_key="<your-provider-api-key>")
```

Run your application with `.venv/bin/python` to use that environment.

## Publishing (maintainers only)

Local installation and build commands above do not publish anything. The separate `Publish to PyPI` workflow in `.github/workflows/publish.yml` runs on pushes to `main` that change `pyproject.toml`, or by manual dispatch. It checks whether the declared version already exists on PyPI; if it does not, it builds with `uv build --out-dir dist` and publishes through the configured PyPI environment. Publishing is a release action, not part of dependency restoration or normal development.
