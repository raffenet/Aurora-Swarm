#!/usr/bin/env python3
"""Wait for vLLM servers to be ready and write a filtered hostfile.

Use on the submit host when vLLM servers are started by a PBS job. The job
creates a hostfile when it starts (which may be hours or days after submit).
This script:

  Phase 1: Waits for the hostfile to appear at the given path and be parseable.
  Phase 2: Polls http://<host>:<port>/health for each endpoint. The health-phase
           timeout starts only after the first healthy node appears. When the
           timeout is reached (or all are up), any still-down nodes are treated
           as failed.
  Output:  Writes a new hostfile containing only endpoints that passed the
           health check. Use this filtered hostfile with scatter_gather_coli.py
           or other clients so failed nodes are omitted.

Example:

  python scripts/wait_for_vllm_servers.py --hostfile /path/to/job_hostfile.txt \\
      --health-timeout 1800 --output /path/to/ready_hostfile.txt

  # Then run your client with the filtered hostfile:
  export AURORA_SWARM_HOSTFILE=/path/to/ready_hostfile.txt
  python examples/scatter_gather_coli.py /path/to/data/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

# Allow running from repo without installing (e.g. python scripts/wait_for_vllm_servers.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aurora_swarm.hostfile import AgentEndpoint, parse_hostfile

HEALTH_PATH = "/health"
DEFAULT_POLL_INTERVAL = 10
DEFAULT_HEALTH_TIMEOUT = 1800  # 30 minutes after first healthy
CONNECT_TIMEOUT = 5


def _configure_logging(quiet: bool) -> logging.Logger:
    logger = logging.getLogger("wait_for_vllm")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG if not quiet else logging.WARNING)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def _endpoint_to_line(ep: AgentEndpoint) -> str:
    parts = [ep.host, str(ep.port)]
    for k, v in sorted(ep.tags.items()):
        parts.append(f"{k}={v}")
    return "\t".join(parts)


def _check_health(
    ep: AgentEndpoint,
    timeout: float = CONNECT_TIMEOUT,
    log: logging.Logger | None = None,
) -> bool:
    url = f"{ep.url}{HEALTH_PATH}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = resp.status == 200
            if log:
                if ok:
                    log.debug("  OK  %s (HTTP %s)", url, resp.status)
                else:
                    log.debug("  FAIL %s (HTTP %s)", url, resp.status)
            return ok
    except Exception as exc:
        if log:
            log.debug("  FAIL %s (%s: %s)", url, type(exc).__name__, exc)
        return False


def _wait_for_hostfile(
    path: Path,
    interval: float,
    timeout: float | None,
    log: logging.Logger,
) -> None:
    """Poll until path exists and is parseable with at least one endpoint."""
    start = time.monotonic()
    while True:
        if path.exists():
            try:
                endpoints = parse_hostfile(path)
                if endpoints:
                    log.info("Hostfile found at %s with %d endpoints", path, len(endpoints))
                    return
                log.debug("Hostfile exists but is empty or has no valid lines, continuing to poll")
            except Exception as e:
                log.debug("Hostfile not yet parseable: %s", e)
        if timeout is not None and (time.monotonic() - start) >= timeout:
            raise SystemExit(f"Hostfile did not appear at {path} within {timeout:.0f}s")
        time.sleep(interval)


def _run_health_phase(
    endpoints: list[AgentEndpoint],
    interval: float,
    health_timeout: float,
    log: logging.Logger,
) -> tuple[list[AgentEndpoint], list[AgentEndpoint]]:
    """Poll /health until all up or health_timeout elapsed since first healthy. Returns (healthy, skipped)."""
    healthy_list: list[AgentEndpoint] = []
    healthy_keys: set[tuple[str, int]] = set()  # (host, port) for O(1) membership
    first_healthy_time: float | None = None
    phase_start = time.monotonic()
    total = len(endpoints)

    while True:
        newly_healthy = 0
        for ep in endpoints:
            if (ep.host, ep.port) in healthy_keys:
                continue
            if _check_health(ep, log=log):
                healthy_keys.add((ep.host, ep.port))
                healthy_list.append(ep)
                newly_healthy += 1
                if first_healthy_time is None:
                    first_healthy_time = time.monotonic()
                    log.info(
                        "First healthy node: %s:%s — health-timeout timer started (%ss)",
                        ep.host,
                        ep.port,
                        health_timeout,
                    )

        if newly_healthy:
            log.info("Healthy %d / %d", len(healthy_list), total)

        if len(healthy_list) == total:
            log.info("All %d vLLM servers are ready.", total)
            return healthy_list, []

        elapsed_since_first = (
            time.monotonic() - first_healthy_time if first_healthy_time is not None else None
        )
        elapsed_since_start = time.monotonic() - phase_start
        if (elapsed_since_first is not None and elapsed_since_first >= health_timeout) or (
            first_healthy_time is None and elapsed_since_start >= health_timeout
        ):
            skipped = [ep for ep in endpoints if (ep.host, ep.port) not in healthy_keys]
            if skipped:
                log.warning(
                    "Health-phase timeout (%.0fs) reached. %d node(s) did not become healthy in time and will be omitted:",
                    health_timeout,
                    len(skipped),
                )
                for ep in skipped:
                    log.warning("  skipped (took too long): %s:%s", ep.host, ep.port)
            if first_healthy_time is None:
                log.error(
                    "No nodes became healthy within %.0fs. Check that endpoints are reachable and serving /health.",
                    health_timeout,
                )
            return healthy_list, skipped

        time.sleep(interval)


def _write_hostfile(path: Path, endpoints: list[AgentEndpoint], log: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ep in endpoints:
            f.write(_endpoint_to_line(ep) + "\n")
    log.info("Wrote filtered hostfile with %d endpoints to %s", len(endpoints), path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wait for vLLM hostfile to appear and servers to be ready; write filtered hostfile.",
        epilog="Use the output hostfile with AURORA_SWARM_HOSTFILE or --hostfile in scatter_gather_coli.py.",
    )
    parser.add_argument(
        "--hostfile",
        type=Path,
        default=None,
        help="Path where the hostfile will be created by the job (or set AURORA_SWARM_HOSTFILE)",
    )
    parser.add_argument(
        "--output",
        "--output-hostfile",
        dest="output",
        type=Path,
        default=None,
        help="Path for filtered hostfile (default: <hostfile>.ready)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Poll interval in seconds (default: %s)" % DEFAULT_POLL_INTERVAL,
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=DEFAULT_HEALTH_TIMEOUT,
        help="Max seconds to wait for remaining servers after first healthy (default: %s)" % DEFAULT_HEALTH_TIMEOUT,
    )
    parser.add_argument(
        "--hostfile-timeout",
        type=float,
        default=None,
        help="Max seconds to wait for hostfile to appear (default: wait indefinitely)",
    )
    parser.add_argument(
        "--min-hosts",
        type=int,
        default=None,
        help="Exit with error if fewer than this many hosts are healthy",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=None,
        help="Exit with error if fraction of healthy hosts is below this (e.g. 0.99)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only log summary messages",
    )
    args = parser.parse_args()

    hostfile_path = args.hostfile
    if hostfile_path is None:
        hostfile_path = os.environ.get("AURORA_SWARM_HOSTFILE")
    if not hostfile_path or not str(hostfile_path).strip():
        parser.error("--hostfile or AURORA_SWARM_HOSTFILE is required")
    hostfile_path = Path(hostfile_path).resolve()

    output_path = args.output
    if output_path is None:
        output_path = hostfile_path.with_suffix(hostfile_path.suffix + ".ready")
    output_path = output_path.resolve()

    log = _configure_logging(args.quiet)

    log.info("Starting wait for vLLM servers (hostfile=%s)", hostfile_path)

    # Phase 1: wait for hostfile
    _wait_for_hostfile(hostfile_path, args.interval, args.hostfile_timeout, log)

    # Re-parse (file may have been updated)
    endpoints = parse_hostfile(hostfile_path)
    if not endpoints:
        raise SystemExit("Hostfile has no valid endpoints.")

    # Phase 2: wait for servers (timeout starts after first healthy)
    healthy_list, skipped = _run_health_phase(endpoints, args.interval, args.health_timeout, log)

    # Min-hosts / min-fraction
    n_healthy = len(healthy_list)
    total = len(endpoints)
    if args.min_hosts is not None and n_healthy < args.min_hosts:
        log.error(
            "Healthy hosts %d is below --min-hosts %d; exiting with error.",
            n_healthy,
            args.min_hosts,
        )
        _write_hostfile(output_path, healthy_list, log)
        raise SystemExit(1)
    if args.min_fraction is not None:
        if total == 0:
            raise SystemExit(1)
        frac = n_healthy / total
        if frac < args.min_fraction:
            log.error(
                "Healthy fraction %.4f is below --min-fraction %s; exiting with error.",
                frac,
                args.min_fraction,
            )
            _write_hostfile(output_path, healthy_list, log)
            raise SystemExit(1)

    _write_hostfile(output_path, healthy_list, log)

    # Final summary
    log.info(
        "Summary: total=%d healthy=%d skipped=%d output=%s",
        total,
        n_healthy,
        len(skipped),
        output_path,
    )


if __name__ == "__main__":
    main()
