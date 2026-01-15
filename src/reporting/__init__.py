"""Reporting module for RL experiment documentation.

Provides tools for generating comprehensive experiment reports
including hypotheses, methods, results, and conclusions.

Modules:
- results_formatter: Results formatting and display
- report_generator: Markdown report generation
"""

from .report_generator import ReportGenerator
from .results_formatter import ResultsFormatter

__all__ = ["ReportGenerator", "ResultsFormatter"]
