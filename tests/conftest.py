"""Shared test configuration — sets up Windows Hadoop env before Spark starts."""

from __future__ import annotations

import os
import sys


def pytest_configure(config):
    """Set HADOOP_HOME and PATH for Windows before any Spark test runs."""
    if sys.platform == "win32":
        os.environ.setdefault("HADOOP_HOME", "C:\\hadoop")
        hadoop_bin = os.path.join(os.environ["HADOOP_HOME"], "bin")
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
        # Ensure Spark uses the correct Python executable
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
