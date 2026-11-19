"""Keep Bokeh models shared between sessions intact when a session is destroyed.

A layout served as a single object (``pn.serve(layout)``, ``layout.show()``, the ICoCo
server, ...) gives the *same* Bokeh figures to every session: a page reload creates a new
session whose document takes the figures (Panel's ``remove_root``), while the previous
session lingers until Bokeh discards it (``unused_session_lifetime``, ~15-30 s).

When it is discarded, ``Document.destroy`` calls ``Model.destroy()`` on every model it
still lists, which sets ``_document = None`` **and clears all the property values** of the
model. The figures the live session is displaying then silently lose their renderers,
data sources and ranges (the "lost renderers"), and their updates are no longer sent.

:func:`protect` makes a document's destruction skip the models that another document
owns. :func:`enable` installs it in the current session and in every new one.
"""

from typing import Any, Optional

from bokeh.document import Document

from scivianna.logging_config import get_logger

logger = get_logger(__name__)

_FLAG = "_scivianna_protected_destroy"
_session_hook_installed = False


def protect(doc: Optional[Document]) -> None:
    """Makes the destruction of ``doc`` leave alone the models owned by another document."""
    if doc is None or getattr(doc, _FLAG, False):
        return
    manager = doc.models

    def destroy() -> None:
        """``DocumentModelManager.destroy`` that skips models taken by another document."""
        kept = 0
        for model in list(manager._models.values()):
            owner = model._document
            if owner is not None and owner is not doc:
                kept += 1
                continue
            model.destroy()
        if kept:
            logger.info("Session document destroyed: kept %d models shared with another session.", kept)
        del manager._models
        del manager._models_by_name

    manager.destroy = destroy
    setattr(doc, _FLAG, True)


def enable() -> None:
    """Protects the current session document and, once, every session created afterwards.

    Session creation hooks can only be registered outside a session; inside one (layout
    built per session), only the current document is protected, which is all it needs.
    """
    global _session_hook_installed
    import panel as pn

    try:
        protect(pn.state.curdoc)
    except Exception:  # no document / not accessible from this thread
        pass
    if _session_hook_installed:
        return
    try:
        pn.state.on_session_created(_on_session_created)
        _session_hook_installed = True
    except RuntimeError:
        pass  # inside a session: objects are per session, the current document is protected


def _on_session_created(session_context: Any) -> None:
    """Session creation hook: protects the new session's document."""
    protect(getattr(session_context, "_document", None))
