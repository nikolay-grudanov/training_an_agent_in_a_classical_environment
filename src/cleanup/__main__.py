"""Entry point for cleanup module when run as: python -m src.cleanup.run"""

from src.cleanup.run import main

if __name__ == "__main__":
    import sys

    sys.exit(main())
