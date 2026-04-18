# scraper/__init__.py

from .pass1 import Phase1Scraper

def __getattr__(name):
    if name == 'Phase2Scraper':
        from .pass2 import Phase2Scraper
        return Phase2Scraper
    raise AttributeError(f"module 'scraper' has no attribute {name!r}")

__all__ = ['Phase1Scraper', 'Phase2Scraper']