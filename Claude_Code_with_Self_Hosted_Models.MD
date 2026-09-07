# Claude Code with Self Hosted Models

## Documentation and Resources 

- [Installation](https://code.claude.com/docs/en/quickstart#native-install-recommended)

## Linux

``` bash
# curl -fsSL https://claude.ai/install.sh | bash

# echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# claude --version
2.1.263 (Claude Code)

# https://nodejs.org/en/download
# https://blog.csdn.net/qq_20236937/article/details/160720814
# npm and node without nvm 
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# npx @playwright/mcp@latest --help
# npx @playwright/mcp@latest --version
# claude mcp add -s user playwright npx '@playwright/mcp@latest'
# claude mcp list
# cat .claude.json | grep playwright
# claude mcp remove -s user playwright

export ANTHROPIC_BASE_URL="http://129.212.179.95:8000"
export ANTHROPIC_AUTH_TOKEN="**********"
export ANTHROPIC_MODEL="Qwen/Qwen3.6-27B"
export ANTHROPIC_SMALL_FAST_MODEL="Qwen/Qwen3.6-27B"
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT="1"

# claude
```

## Windows

``` bash
> irm https://claude.ai/install.ps1 | iex

> claude --version
2.1.263 (Claude Code)

# nvm, npm and node

# npx @playwright/mcp@latest --help
# npx @playwright/mcp@latest --version
# claude mcp add -s user playwright npx '@playwright/mcp@latest'
# claude mcp list
# cat .claude.json | grep playwright
# claude mcp remove -s user playwright

$env:ANTHROPIC_BASE_URL="http://129.212.179.95:8000"
$env:ANTHROPIC_AUTH_TOKEN="**********"
$env:ANTHROPIC_MODEL="Qwen/Qwen3.6-27B"
$env:ANTHROPIC_SMALL_FAST_MODEL="Qwen/Qwen3.6-27B"
$env:CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT="1"

> claude
```
