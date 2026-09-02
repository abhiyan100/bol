"""`python -m bol.wake`: the keyword listener child process."""

import sys

from .listener import main

if __name__ == "__main__":
    sys.exit(main())
