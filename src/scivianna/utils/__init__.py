_testing: bool = False
"""If True, scivianna is being used in a testing context, callbacks are immediatly called.

If False, callbacks are called on next tick: it prevents conflicts between threads."""