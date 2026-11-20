from pathlib import Path
import os
import nbclient
import nbformat
from traitlets.config import Config

def execute_notebook(
        nb_path: Path,
        version: object = 4,
        timeout: object = 600,
        kernel_name: object = 'python3') -> None:

    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=version)

    os.environ['CURRENT_NOTEBOOK'] = str(nb_path)

    client = nbclient.NotebookClient(
        nb, timeout=timeout, kernel_name=kernel_name,
        config=Config({'KernelManager': {'transport': 'tcp'}}))
    client.execute()