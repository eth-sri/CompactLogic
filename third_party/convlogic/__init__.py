"""Minimal ConvLogic adapter/report support for the shared compiler/simulation stack."""

__all__ = ['generate_convlogic_paper_stats']


def generate_convlogic_paper_stats(*args, **kwargs):
    from .paper_stats import generate_convlogic_paper_stats as _impl

    return _impl(*args, **kwargs)
