"""Entry point for audit module when run as: python -m src.audit.run"""

from src.audit.run import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
