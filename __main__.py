# arxiv_agent/__main__.py

import os
import sys

import streamlit.web.cli as stcli
from . import app


def main() -> None:
    """
    Entry point for the `arxiv-agent` console script.

    It programmatically calls `streamlit run` on the packaged app.py file.
    """
    script_path = os.path.abspath(app.__file__)
    sys.argv = ["streamlit", "run", script_path]
    stcli.main()


if __name__ == "__main__":
    main()

