"""Debug logging of what Bokeh sends to the browser and what the browser fails to apply.

Enabled with the environment variable ``SCIVIANNA_DEBUG_RENDER=1``; does nothing otherwise.

* Python side (terminal, logger ``scivianna.utils.render_debug``): every document change that
  creates or moves a Legend, or changes a plot's side panels (above/below/left/right/center)
  or renderers, with the scivianna code line that caused it.
* Browser side (console, prefix ``[sv-debug]``): every patch received from the server
  (models changed, new models), errors raised while applying it, and, when a plot fails to
  lay out (``cannot find a view for ...`` / ``reading 'top_panel'``), which plot it is and
  which of its panel models have no view / are unknown to the browser.

Call :func:`enable_if_requested` once (done by the plotters); it installs the probes in the
current session document and in every new session.
"""

import os
import traceback
from typing import Any, Optional

from bokeh.document import Document
from bokeh.document.events import ModelChangedEvent
from bokeh.model import Model
from bokeh.models import ColumnDataSource, CustomJS

from scivianna.logging_config import get_logger

logger = get_logger(__name__)

ENV_VAR = "SCIVIANNA_DEBUG_RENDER"
PANEL_ATTRS = {"above", "below", "left", "right", "center", "renderers"}
_DOC_FLAG = "_scivianna_render_debug"
_session_hook_installed = False


def enabled() -> bool:
    """Whether render debugging is requested through ``SCIVIANNA_DEBUG_RENDER``."""
    return os.environ.get(ENV_VAR, "").lower() not in ("", "0", "false", "no")


def log(message: str, *args: Any) -> None:
    """Logs a render-debug message (only when enabled)."""
    if enabled():
        logger.warning("[render-debug] " + message, *args)


def describe(model: Any) -> str:
    """Short ``Type(id, name)`` description of a Bokeh model (or the value itself)."""
    if isinstance(model, Model):
        name = getattr(model, "name", None)
        return f"{type(model).__name__}({model.id}{', ' + repr(name) if name else ''})"
    return repr(model)[:80]


def _models_in(value: Any) -> list:
    """Bokeh models directly held by a property value (a model or a list of models)."""
    if isinstance(value, Model):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, Model)]
    return []


def _caller() -> str:
    """The innermost scivianna / scivianna_graph frames of the current stack, outside this module."""
    frames = [
        f for f in traceback.extract_stack()[:-3]
        if ("scivianna" in f.filename) and "render_debug" not in f.filename
    ]
    return " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in reversed(frames[-4:])) or "?"


def _placement(legend: Model, roots: list) -> str:
    """Where a legend sits: ``figure.side`` of the plot holding it among the given models' references."""
    for root in roots:
        for plot in root.references():
            for side in ("above", "below", "left", "right", "center"):
                value = getattr(plot, side, None)
                if isinstance(value, (list, tuple)) and any(v is legend for v in value):
                    return f"{describe(plot)}.{side}"
    return "?"


def _on_change(event: Any) -> None:
    """Document change listener: logs legend / plot-panel changes with their origin (never raises)."""
    try:
        _log_change(event)
    except Exception as e:  # debugging must not break the application
        log("change logger failed: %s: %s", type(e).__name__, e)


def _log_change(event: Any) -> None:
    """Logs legend / plot-panel changes of one document event."""
    if not isinstance(event, ModelChangedEvent):
        return
    model, attr = event.model, event.attr
    if getattr(model, "name", None) == "scivianna_render_debug_sink":
        return
    new_models = _models_in(event.new)
    legends = [m for m in new_models if type(m).__name__ in ("Legend", "LegendItem")]
    # Legends brought in by a whole new sub-tree (e.g. a new figure or pane) do not get
    # their own change event: look for them inside the new models.
    seen = event.document.__dict__.setdefault("_scivianna_render_debug_legends", set())
    nested = []
    for m in new_models:
        for ref in m.references():
            if type(ref).__name__ == "Legend" and ref not in legends and ref.id not in seen:
                seen.add(ref.id)
                nested.append(ref)
    for legend in nested:
        log(
            "doc %s: %s.%s brings %s (items %s) placed in %s  (from %s)",
            id(event.document) % 10000, describe(model), attr, describe(legend),
            [i.label.value if i.label is not None else None for i in legend.items],
            _placement(legend, new_models), _caller(),
        )
    if attr in PANEL_ATTRS or legends or type(model).__name__ in ("Legend", "LegendItem"):
        log(
            "doc %s: %s.%s = [%s]  (from %s)",
            id(event.document) % 10000, describe(model), attr,
            ", ".join(describe(m) for m in new_models) if new_models else describe(event.new),
            _caller(),
        )


BROWSER_JS = r"""
if (!window.__sv_render_debug) {
  window.__sv_render_debug = true;
  const P = "[sv-debug]";
  // Messages are also sent to Python (logged in the terminal) through the sink data source.
  const queue = [];
  const flush = () => {
    if (!queue.length) return;
    const lines = queue.splice(0);
    try { sink.data = {level: lines.map((l) => l[0]), text: lines.map((l) => l[1])}; } catch (err) { }
  };
  setInterval(flush, 300);
  const out = (level) => (...parts) => {
    console[level](...parts);
    if (level !== "debug") queue.push([level, parts.map((x) => typeof x === "string" ? x : String(x)).join(" ")]);
  };
  const LOG = {warn: out("warn"), error: out("error"), debug: out("debug")};
  const docs = () => (window.Bokeh && Bokeh.documents) ? Bokeh.documents : [];
  const find = (id) => { for (const d of docs()) { const m = d.get_model_by_id(id); if (m) return m; } return null; };
  const desc = (m) => m ? `${m.type}(${m.id}${m.name ? ", " + m.name : ""})` : "?";
  const allViews = () => {
    const out = new Map();
    const walk = (v) => {
      if (!v || out.has(v)) return; out.set(v, true);
      for (const c of (v.child_views ?? [])) walk(c);
      if (v.renderer_views) for (const r of v.renderer_views.values()) walk(r);
      if (v._child_views) for (const r of v._child_views.values()) walk(r);
    };
    for (const r of Object.values(Bokeh.index ?? {})) walk(r);
    return [...out.keys()];
  };
  const hasView = (id) => allViews().some((v) => v.model && v.model.id === id);
  const plotOf = (id) => {
    for (const d of docs()) for (const m of d._all_models.values()) {
      for (const side of ["above", "below", "left", "right", "center", "renderers"]) {
        const arr = m[side];
        if (Array.isArray(arr) && arr.some((x) => x && x.id === id)) return `${desc(m)}.${side}`;
      }
    }
    return "no plot";
  };
  const explain = (text) => {
    const ids = [...new Set(String(text).match(/\bp\d+\b/g) ?? [])];
    for (const id of ids) {
      const m = find(id);
      LOG.warn(P, `  ${id}: ${m ? desc(m) : "UNKNOWN to the browser"}, view: ${hasView(id)}, held by: ${m ? plotOf(id) : "-"}`);
    }
  };
  window.addEventListener("error", (e) => { LOG.warn(P, "window error:", e.message); explain(e.message); });
  window.addEventListener("unhandledrejection", (e) => { const msg = e.reason?.message ?? String(e.reason); LOG.warn(P, "unhandled rejection:", msg); explain(msg); });

  // Patches received from the server.
  let n = 0;
  const newDefs = (value, out) => {
    if (value == null || typeof value !== "object") return out;
    if (Array.isArray(value)) { for (const v of value) newDefs(v, out); return out; }
    if (value.type === "object" && value.id && value.name) out.push(`${value.name}(${value.id})`);
    for (const v of Object.values(value)) newDefs(v, out);
    return out;
  };
  const patchDoc = (doc) => {
    const proto = Object.getPrototypeOf(doc);
    if (proto.__sv_patched) return; proto.__sv_patched = true;
    const orig = proto.apply_json_patch;
    proto.apply_json_patch = function (patch, buffers) {
      const i = ++n;
      const events = (patch.events ?? []).map((e) => {
        const target = e.model?.id ? desc(this.get_model_by_id(e.model.id)) + (this.get_model_by_id(e.model.id) ? "" : `[${e.model.id} UNKNOWN]`) : "";
        return `${e.kind} ${target}${e.attr ? "." + e.attr : ""}`;
      });
      const created = newDefs(patch.events ?? [], []);
      const interesting = events.some((e) => /Legend|\.(above|below|left|right|center|renderers)\b/.test(e)) || created.length;
      (interesting ? LOG.warn : LOG.debug)(P, `patch #${i}:`, events.join(" | "), created.length ? "\n   new: " + created.join(", ") : "");
      try {
        return orig.call(this, patch, buffers);
      } catch (err) {
        LOG.error(P, `patch #${i} FAILED:`, err.message); explain(err.message); throw err;
      }
    };
    LOG.warn(P, "patch logging installed");
  };

  // Plot layout failures: name the plot and the side-panel models without a view.
  const patchPlotView = () => {
    for (const v of allViews()) {
      if (!v.model || !("frame_width" in v.model) || !("above" in v.model)) continue;
      let proto = Object.getPrototypeOf(v);
      while (proto && !Object.prototype.hasOwnProperty.call(proto, "_update_layout")) proto = Object.getPrototypeOf(proto);
      if (!proto || proto.__sv_patched) return true;
      proto.__sv_patched = true;
      const orig = proto._update_layout;
      proto._update_layout = function () {
        try { return orig.call(this); }
        catch (err) {
          const m = this.model;
          LOG.error(P, `layout of ${desc(m)} FAILED: ${err.message}`);
          for (const side of ["above", "below", "left", "right", "center"]) {
            for (const x of (m[side] ?? [])) {
              const known = !!find(x.id);
              const view = this.views?.find_one ? this.views.find_one(x) : (this.renderer_views?.get?.(x));
              if (!known || view == null) LOG.error(P, `   ${side}: ${desc(x)} known: ${known}, view: ${view != null}`);
            }
          }
          throw err;
        }
      };
      LOG.warn(P, "plot layout logging installed");
      return true;
    }
    return false;
  };
  LOG.warn(P, "probe running in the browser");
  const install = () => { for (const d of docs()) patchDoc(d); return patchPlotView(); };
  let tries = 0;
  const timer = setInterval(() => { if (install() || ++tries > 120) clearInterval(timer); }, 500);
}
"""


def _on_browser_message(attr: str, old: Any, new: Any) -> None:
    """Logs the messages the browser probe sends through the sink data source."""
    for level, text in zip(new.get("level", []), new.get("text", [])):
        log("BROWSER %s: %s", level, text)


def install(doc: Optional[Document]) -> None:
    """Installs the Python change logger and the browser probe in a document (once)."""
    if doc is None or not enabled() or getattr(doc, _DOC_FLAG, False):
        return
    setattr(doc, _DOC_FLAG, True)
    doc.on_change(_on_change)
    sink = ColumnDataSource(data={"level": [], "text": []}, name="scivianna_render_debug_sink")
    sink.on_change("data", _on_browser_message)
    doc.add_root(sink)
    doc.js_on_event("document_ready", CustomJS(code=BROWSER_JS, args={"sink": sink}))
    log("installed in document %s", id(doc) % 10000)


def enable_if_requested() -> None:
    """When ``SCIVIANNA_DEBUG_RENDER`` is set, installs the probes in the current and every new session."""
    global _session_hook_installed
    if not enabled():
        return
    import panel as pn

    try:
        install(pn.state.curdoc)
    except Exception:  # no session / document locked: the session hook covers it
        pass
    if not _session_hook_installed:
        try:
            pn.state.on_session_created(lambda ctx: install(getattr(ctx, "_document", None)))
            _session_hook_installed = True
        except RuntimeError:
            pass  # inside a session (layout built per session): the current document is enough
