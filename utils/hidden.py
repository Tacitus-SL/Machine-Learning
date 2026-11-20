import os
import sys
import importlib
import typing
import re
from pathlib import Path

try:
    import source
except ImportError:
    source = None

def get_running_notebook_path() -> typing.Optional[str]:
    """Ask the local Jupyter server which notebook file owns this kernel."""
    try:
        import ipykernel
        connection_file = Path(ipykernel.get_connection_file()).name
        kernel_id = re.search(r"kernel-(.*)\.json", connection_file).group(1)
    except Exception:
        return None

    try:
        from jupyter_server import serverapp as jupyter_serverapp
        servers = list(jupyter_serverapp.list_running_servers())
    except ImportError:
        try:
            from notebook import notebookapp as jupyter_serverapp
            servers = list(jupyter_serverapp.list_running_servers())
        except ImportError:
            servers = []

    for server in servers:
        try:
            url = server["url"] + "api/sessions"
            if server.get("token"):
                url += "?token=" + server["token"]
            with urllib.request.urlopen(url, timeout=2) as response:
                sessions = json.load(response)
            for session in sessions:
                if session["kernel"]["id"] == kernel_id:
                    return session["notebook"]["path"]
        except Exception:
            continue
    return None

def detect() -> bool:
    """Return whether the calling notebook or Python script is hidden.

    A notebook or script is hidden when its filename starts with a dot.

    Returns
    -------
    bool
        ``True`` when the caller is a hidden notebook or Python script.

    Examples
    --------
    Calling ``detect()`` from ``.task.py`` returns ``True``, while calling it
    from ``task.py`` returns ``False``.
    """
    notebook_name = (
        os.environ.get('CURRENT_NOTEBOOK')
        or globals().get('__vsc_ipynb_file__')
    )

    if notebook_name:
        return Path(notebook_name).name.startswith('.')

    caller_frame = sys._getframe(1)
    script_name = caller_frame.f_globals.get("__file__")
    if script_name:
        return Path(script_name).name.startswith(".")

    caller_path = Path(caller_frame.f_code.co_filename)
    if caller_path.suffix == ".py":
        return caller_path.name.startswith(".")

    notebook_name = get_running_notebook_path()
    return bool(notebook_name and Path(notebook_name).name.startswith("."))

def replace_code() -> None:
    if source is None:
        return

    src_dir = Path(source.__file__).resolve().parent
    hidden_dir = src_dir / ".code"
    init_file = hidden_dir / "__init__.py"

    # Remove the normal source.code package and all its imported submodules.
    for module_name in list(sys.modules):
        if module_name == "source.code" or module_name.startswith("source.code."):
            del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        "source.code",
        init_file,
        submodule_search_locations=[str(hidden_dir)],
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load hidden package from {hidden_dir}")

    hidden_code = importlib.util.module_from_spec(spec)

    # Register before executing so relative imports inside .code work.
    sys.modules["source.code"] = hidden_code
    source.code = hidden_code

    try:
        spec.loader.exec_module(hidden_code)
    except Exception:
        sys.modules.pop("source.code", None)
        raise
