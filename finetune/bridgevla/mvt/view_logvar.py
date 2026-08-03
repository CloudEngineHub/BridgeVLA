"""Placeholder for the Kendall-Gal per-view uncertainty weighting branch.

An adaptive per-view ``log σ²`` weighting (instead of the uniform 1/N
per-view averaging the model uses today) needs a per-view feature source
on the current PaliGemma-direct architecture (e.g. spatial mean of
PaliGemma patches) and is left as future work.

When ``cfg.use_view_logvar=True``, the model constructor raises
``NotImplementedError`` so the deferred branch fails fast rather than
silently doing the wrong thing.
"""


_DEFERRED_MSG = (
    "use_view_logvar=True branch is deferred; implement a per-view feature "
    "source on PaliGemma-direct features before re-enabling this flag."
)


def assert_view_logvar_disabled(use_view_logvar: bool) -> None:
    """Call from model constructors to fail fast when the flag is True."""
    if use_view_logvar:
        raise NotImplementedError(_DEFERRED_MSG)
