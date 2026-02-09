"""CLI entry point for mcp-python-sandbox."""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mcp-python-sandbox",
        description="MCP server for secure Python script execution",
    )
    parser.add_argument(
        "--python-version",
        default="3.13",
        help="Python version for script execution (default: 3.13)",
    )
    parser.add_argument(
        "--sandbox-backend",
        choices=["native", "docker", "none"],
        default="native",
        help="Sandbox backend: native (bwrap/sandbox-exec), docker, or none (default: native)",
    )
    parser.add_argument(
        "--max-timeout",
        type=int,
        default=300,
        help="Maximum allowed timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--default-timeout",
        type=int,
        default=30,
        help="Default timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=102_400,
        help="Maximum output size in bytes (default: 102400)",
    )
    parser.add_argument(
        "--no-warm-cache",
        action="store_true",
        help="Skip cache warming on startup",
    )
    parser.add_argument(
        "--uv-path",
        default="uv",
        help="Path to uv binary (default: uv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from mcp_python_sandbox.config import ServerConfig
    from mcp_python_sandbox.server import create_server

    config = ServerConfig(
        python_version=args.python_version,
        sandbox_backend=args.sandbox_backend,
        max_timeout=args.max_timeout,
        default_timeout=args.default_timeout,
        max_output_bytes=args.max_output_bytes,
        warm_cache=not args.no_warm_cache,
        uv_path=args.uv_path,
    )

    server = create_server(config)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
