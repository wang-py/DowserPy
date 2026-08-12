"""
dowser_run.py

Subprocess helpers for driving the DowserPy binaries.

Two things make these binaries awkward to call directly:

  * allwaters and montecarlo prompt on stdin and block until answered, so they
    have to be fed a scripted set of answers rather than run to completion.

  * allwaters exits 1 even on a completely successful run, so its exit status
    says nothing about whether it worked. Stages that call it check for their
    expected output files instead of trusting the return code.

Both binaries also draw progress with carriage returns and ANSI escapes; their
output is captured to a log file rather than streamed to the terminal.
"""

import os
import subprocess


class StageError(RuntimeError):
    """A pipeline stage failed to produce what it promised."""


def run(argv, cwd, log_path=None, stdout_path=None, check=True):
    """
    Run `argv` in `cwd`.

    If `stdout_path` is given the command's stdout is written there (used for
    the tools that report results on stdout); otherwise stdout joins stderr in
    the log. Returns the exit status.
    """
    log = open(log_path, "wb") if log_path else subprocess.DEVNULL
    out = open(stdout_path, "wb") if stdout_path else log

    try:
        status = subprocess.call(argv, cwd=cwd, stdout=out, stderr=log)
    finally:
        for handle in (out, log):
            if handle not in (subprocess.DEVNULL, None) and not handle.closed:
                handle.close()

    if check and status != 0:
        raise StageError(
            "%s exited %d (see %s)" % (os.path.basename(argv[0]), status, log_path)
        )
    return status


def run_interactive(argv, cwd, answers, log_path=None, timeout=None):
    """
    Run `argv` in `cwd`, writing `answers` to its stdin.

    `answers` is a list of strings, one per prompt, written up front in order;
    the binaries here read their prompts sequentially and never branch on
    earlier answers, so there is no need to wait for each prompt individually.

    The exit status is returned rather than checked: allwaters exits 1 on
    success, so callers verify their output files instead.
    """
    log = open(log_path, "wb") if log_path else subprocess.DEVNULL
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
        )
        process.communicate(
            input="".join(answer + "\n" for answer in answers).encode(),
            timeout=timeout,
        )
    finally:
        if log not in (subprocess.DEVNULL, None) and not log.closed:
            log.close()

    return process.returncode


def require(path, what, log_path=None):
    """Raise StageError unless `path` exists and is non-empty."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        message = "%s was not produced (expected %s)" % (what, path)
        if log_path:
            message += "; see %s" % log_path
        raise StageError(message)
    return path
