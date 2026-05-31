# Colab MCP Transport Issue

## Summary

The Colab MCP tools are currently unavailable in this Codex session because the
MCP transport for the `colab` server is closed. This blocks tool calls such as
opening the Colab connection URL, checking Colab status, running cells, or
starting notebook jobs from Codex.

This does **not** block the HippoGraph repo itself. The neighborhood transformer
training code has been added and pushed, but Colab cannot be controlled through
the MCP tools until the bridge/client transport is reconnected.

## Observed Errors

These MCP tool calls failed:

```text
mcp__colab__.colab_connect
mcp__colab__.colab_status
mcp__colab__.colab_connection_url
functions.list_mcp_resources(server="colab")
```

They all returned the same underlying failure:

```text
Transport closed
```

Example:

```text
tool call error: tool call failed for `colab/colab_connect`

Caused by:
    Transport closed
```

## What Was Tried

Checked for an existing local adapter process:

```bash
ps -eo pid,cmd | rg 'colab|codex-adapter'
```

No active Colab adapter process was visible.

Tried starting the adapter manually from:

```text
/home/cameron/projects/google-collab-codex-con
```

Command:

```bash
uv --cache-dir /tmp/uv-cache run colab-codex-adapter
```

The adapter started a FastMCP stdio server:

```text
Server: ColabCodexAdapter
Starting MCP server 'ColabCodexAdapter' with transport 'stdio'
```

But subsequent Codex MCP tool calls still failed with:

```text
Transport closed
```

## Likely Cause

The Codex session's MCP client-side connection to the `colab` server is stale or
closed. Running the adapter command manually is not enough to repair the already
closed MCP tool transport, because the adapter is a stdio MCP server that needs
to be launched and attached by the MCP host/client process.

In short:

```text
The adapter code can start, but Codex's registered colab MCP transport is closed.
```

## Expected Fix

Restart or reattach the Colab MCP server from the host/client side, not from
inside this repo shell.

Likely recovery actions:

1. Restart the Codex client/session MCP server configuration.
2. Ensure the `colab` MCP server is launched by the MCP host with stdio attached.
3. Retry:

```text
mcp__colab__.colab_status
mcp__colab__.colab_connect
```

A healthy connection should allow:

```text
mcp__colab__.colab_connection_url
mcp__colab__.colab_run_python
mcp__colab__.colab_run_python_async
```

## Current Repo State

The repo is ready for Colab training once MCP is restored.

Useful commands after Colab is available:

```bash
python3 python/synthetic/generate.py \
  --output data/synthetic/librarian_cases.jsonl \
  --count 5000 \
  --candidates 32

python3 -m python.training.train_librarian \
  --dataset data/synthetic/librarian_cases.jsonl \
  --output artifacts/librarian/neighborhood_transformer.pt \
  --epochs 8
```

Latest pushed commit containing the transformer training stack:

```text
37ccbcd Add neighborhood transformer training stack
```

