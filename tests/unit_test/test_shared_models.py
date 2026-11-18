"""Bokeh models shared between sessions survive the destruction of a stale session.

A layout served as one object gives the same figures to every session; a reload makes
the new session's document take them. When Bokeh later destroys the old session,
``Model.destroy()`` clears the property values of every model the old document lists:
the live figures lost their renderers. ``shared_models.protect`` prevents that.
"""

import panel as pn
from bokeh.document import Document
from bokeh.plotting import figure

from scivianna.utils import shared_models


def _two_sessions():
    """A figure shown by a first document, then taken by a second one (page reload)."""
    fig = figure(width=100, height=100)
    fig.line([0, 1], [0, 1])
    pane = pn.pane.Bokeh(fig)
    old, new = Document(), Document()
    old.add_root(pn.Column(pane).get_root(old))
    new.add_root(pn.Column(pane).get_root(new))
    assert fig.document is new
    return fig, old, new


def _destroy(doc):
    """Destroys a document the way Bokeh does when it discards its session."""
    def session(event):
        """Change receiver standing for the server session."""

    doc.on_change(session)
    doc.destroy(session)


def test_unprotected_destroy_wipes_shared_figure():
    """The Bokeh behaviour this module works around (if Bokeh fixes it, protect becomes a no-op)."""
    fig, old, _ = _two_sessions()
    _destroy(old)
    assert len(fig.renderers) == 0


def test_protected_destroy_keeps_shared_figure():
    """Models owned by the live document keep their values and their document."""
    fig, old, new = _two_sessions()
    shared_models.protect(old)
    _destroy(old)
    assert len(fig.renderers) == 1
    assert fig.document is new


def test_protected_destroy_still_destroys_own_models():
    """Models that only belong to the destroyed document are destroyed as before."""
    doc = Document()
    fig = figure()
    fig.line([0, 1], [0, 1])
    doc.add_root(fig)
    shared_models.protect(doc)
    _destroy(doc)
    assert fig.document is None
    assert len(fig.renderers) == 0
