import sys

def call_callback_func(code, offset, func=None, arg0=None):
    if func:
        print(f"Function {func.__name__} called")

def return_callback_func(code, offset, retval=None):
    print(f"Function {code.co_name} returned")

# tool_id = 0
tool_id = sys.monitoring.DEBUGGER_ID

sys.monitoring.use_tool_id(tool_id, "SimpleMonitor")
sys.monitoring.register_callback(tool_id, sys.monitoring.events.CALL, call_callback_func)
sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_RETURN, return_callback_func)
sys.monitoring.set_events(tool_id, sys.monitoring.events.CALL | sys.monitoring.events.PY_RETURN)

def bar():
    return

def foo():
    bar()
    return

foo()
sys.monitoring.free_tool_id(tool_id)