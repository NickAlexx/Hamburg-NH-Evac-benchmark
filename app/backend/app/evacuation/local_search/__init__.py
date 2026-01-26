# Path: backend/app/evacuation/local_search/__init__.py

"""
Local search package for the evacuation solver.

Exposes:
- MemeticImprover: the memetic/VND/ALNS local-search engine.
"""

from .memetic import MemeticImprover

__all__ = ["MemeticImprover"]
