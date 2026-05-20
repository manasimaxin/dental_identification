# nvm Project Instructions (Gemini)

This document provides foundational mandates and guidance for the Gemini CLI agent when working with the Node Version Manager (nvm) codebase. These instructions take precedence over general defaults.

## Core Mandates

- **Portability First**: nvm is a POSIX-compliant shell function. All changes must be compatible with `sh`, `dash`, `bash`, `ksh`, and `zsh`.
- **Testing Requirement**: Every change MUST be verified with the `urchin` test suite across multiple shells.
- **ShellCheck Compliance**: All shell scripts must pass `shellcheck` with appropriate directives where necessary.

## Project Overview

nvm is a version manager for Node.js, implemented primarily in shell script.

### Key Files
- `nvm.sh`: Core functionality and main `nvm()` function.
- `install.sh`: Installation script.
- `nvm-exec`: Execution wrapper for running commands with specific Node versions.
- `bash_completion`: Tab completion for bash.

## Development Workflows

### Testing
Use the [urchin](https://www.npmjs.com/package/urchin) test framework.

- **Run all tests**: `npm test` or `make test`
- **Run specific suite**: `make TEST_SUITE=fast test`
- **Run in specific shell**: `make SHELLS=bash test`
- **Individual test (safe)**: `./node_modules/.bin/urchin 'test/path/to/test'`

### Linting
Use `shellcheck`.
- `shellcheck -s bash nvm.sh`
- Common directives:
  - `# shellcheck disable=SC2039`: Allow bash extensions in POSIX mode (where handled).
  - `# shellcheck disable=SC3043`: Allow `local` keyword.

## Coding Standards

- **Indentation**: 2 spaces.
- **Naming**: Internal functions must be prefixed with `nvm_`.
- **Output**: Use `nvm_echo` and `nvm_err` instead of `echo`.
- **Variables**: Always quote variables: `"${VAR}"`.
- **Locals**: Declare `local FOO` and then initialize on the next line for `ksh` compatibility.

## Architectural Patterns

- **Lazy Loading**: Implement lazy loading for optional features.
- **Caching**: Use `$NVM_DIR/.cache` for expensive operations.
- **No Subprocesses**: Minimize subprocess calls to maintain performance.

---
*Derived from CLAUDE.md*
