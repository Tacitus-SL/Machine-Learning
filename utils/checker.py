import numbers
import os
import re
import shutil
import tempfile
import numpy
import typing
try:
    import torch
except ImportError:
    torch = None

from pathlib import Path
import numpy as np
import importlib.util
import unittest
import sys

from . import nb_executor

__unittest = True  # tells unittest to strip this module's frames from failure tracebacks


def _patch_teamcity_traceback_filter():
    """
    PyCharm's own test runner (_jb_unittest_runner.py) does not use plain
    unittest's failure formatting -- it uses the bundled 'teamcity-messages'
    package (teamcity/unittestpy.py), whose TeamcityTestResult.report_fail
    builds failure text via:

        convert_error_to_string(err, ["diff_tools", "case.py"])

    That's a hardcoded filename allowlist with no extension point, so this
    module's own make_test() frame always shows up in PyCharm's failure
    panel, unlike under plain `python -m unittest` (handled above via
    __unittest). This patches the one lookup report_fail relies on so it
    also skips this file, the same way it already skips unittest's case.py.
    No-ops when not running under PyCharm's runner.
    """
    try:
        from teamcity import unittestpy
    except ImportError:
        return

    original_convert = unittestpy.convert_error_to_string
    if getattr(original_convert, "_checker_patched", False):
        return  # already patched, e.g. if this module got imported twice

    this_file = os.path.basename(__file__)

    def convert_and_skip_this_file(err, frames_to_skip_from_tail=None):
        skip = list(frames_to_skip_from_tail or [])
        if this_file not in skip:
            skip.append(this_file)
        text = original_convert(err, skip)

        # convert_error_to_string only filters individual frame lines; the
        # "Traceback (most recent call last):" header is unconditional, so
        # it survives even when every frame under it got filtered out.
        # Drop it too, but only if no frame actually remains -- a real
        # traceback (e.g. a genuine bug elsewhere) should still show it.
        lines = text.splitlines(keepends=True)
        if lines and lines[0].startswith("Traceback (most recent call last)") \
                and not any(line.startswith("  File ") for line in lines[1:]):
            lines = lines[1:]
            # No real traceback survived -- this is one of our own
            # synthetic status messages (see make_test below), raised as
            # a bare AssertionError(status). The leading "AssertionError: "
            # type-prefix adds nothing there, so drop it too.
            if lines:
                lines[0] = re.sub(r'^\w[\w.]*:\s*', '', lines[0], count=1)
        return "".join(lines)

    convert_and_skip_this_file._checker_patched = True
    unittestpy.convert_error_to_string = convert_and_skip_this_file


_patch_teamcity_traceback_filter()

ANSWER_KEY_SEPARATOR = " -> "


def normalize_answer(input_dict: typing.Dict[str, typing.Any], sep: str = ' -> ') -> typing.Dict[str, np.ndarray|None]:
    """
    Flattens dictionary recursively + turns all values to numpy arrays, example:
    input: {'a': {'b': 1, 'c': 2}, 'd': 3}
    output: {'a -> b': np.array(1), 'a -> c': np.array(2), 'd': np.array(3)}

    The flattened dict contains only numpy arrays or None.

    :param input_dict: Dictionary with input values
    :param sep: Separator to use for flattening

    :return: flattened dictionary
    """

    flattened = {}

    def flatten(current, prefix=""):
        if isinstance(current, dict):
            if not current:
                flattened[prefix] = np.array([])
                return
            for key, value in current.items():
                next_key = str(key) if not prefix else f"{prefix}{sep}{key}"
                flatten(value, next_key)
            return

        if isinstance(current, (list, tuple)):
            if not current:
                flattened[prefix] = np.array(current)
                return
            for index, value in enumerate(current):
                next_key = str(index) if not prefix else f"{prefix}{sep}{index}"
                flatten(value, next_key)
            return

        if current is not None:
            flattened[prefix] = np.array(current)
        else:
            flattened[prefix] = None

    flatten(input_dict)
    return flattened


def save_npz(answer: typing.Dict[str, typing.Any], filename: str | Path):
    """
    Saves answer in the form of dictionary to npz file

    :param answer: Dictionary with answer
    :param filename: String, speicfying path to npz file
    :return: None
    """
    normalized = normalize_answer(answer)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **normalized)


def load_answer(
        filename: str|Path,
        default: typing.Any = None,
        missing_ok: bool = False) -> typing.Dict[str, typing.Any]:
    """
    Loads answer from npz file

    :param filename: String, specifying path to npz file
    :param default: Value to return if file does not exist
    :param missing_ok: Whether it is a problem if the file does not exist
    :return:
    """
    path = Path(filename)
    if missing_ok and not path.exists():
        return {}
    return _load_npz(str(path))


def check_condition(condition, info, successes, failures, test_weight):
    if condition:
        successes.append((info, 'PASSED', test_weight))
    else:
        failures.append((info, 'FAILED', test_weight))


def get(collection, key):
    res = None
    message = None
    try:
        res = collection[key]
    except:
        res = None

    return res


def run_checks(
        received: typing.Dict[str, typing.Any],
        expected: typing.Dict[str, typing.Any],
        item_path: typing.Union[str, typing.Tuple[str, ...]] = (),
        test_info: str = 'task',
        eps: float =1.0e-4,
        report_fails_only:bool = True,
        include_explanations:bool = True):

    def _build_status(name, ok, status):
        entry = {'name': name, 'ok': ok}
        if include_explanations:
            entry['status'] = status
        return entry

    if expected is None:
        if received is None:
            if not report_fails_only:
                return [_build_status(test_info, True, "OK")]
            else:
                return []
        else:
            return [_build_status(test_info, False, f"Expecting None, got {str(received)}")]

    expected = normalize_answer(expected)
    received = normalize_answer(received)

    if isinstance(item_path, str):
        item_path = (item_path,)

    if len(item_path) > 0:
        joint_key = ANSWER_KEY_SEPARATOR.join(item_path)
        prefix = joint_key + ANSWER_KEY_SEPARATOR

        # item_path names a prefix, not necessarily an already-flat leaf:
        # matches the key itself (the leaf case) plus every key nested
        # under it (the group case, e.g. '1D' matching '1D -> d1', '1D -> dd1', ...).
        matching = {key: value for key, value in expected.items() if key == joint_key or key.startswith(prefix)}

        if not matching:
            raise ValueError(f"item_path {item_path!r} invalid for the teacher's answer: no key equal to or prefixed by {joint_key!r} in {expected!r}")

        expected = matching


    report = []
    for key in expected:
        fail = False
        if key not in received:
            report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, f'FAILED. Item is not found (missing {key!r} in your answer)'))
            continue

        if expected[key] is None:
            if received[key] is not None:
                report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, f'FAILED. Expected None, received {received[key]!r}'))
                fail = True

        # Checking if Numpy array is empty
        if isinstance(expected[key], np.ndarray):
            if not isinstance(received[key], np.ndarray):
                report.append(_build_status(
                    test_info + ANSWER_KEY_SEPARATOR + key, False,
                    f'FAILED. Expected numpy array, received {type(received[key])} instead'
                ))
                fail = True
            elif expected[key].size == 0:
                if received[key].size != 0:
                    explanation = (
                        'FAILED: expected an empty tensor, got non-empty one:\n' +
                        'Expected size: {}\n'.format(str(expected[key].size)) +
                        'Received size: {}\n'.format(str(received[key].size))
                    )
                    report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, explanation))
                    fail = True


            # Checking if Numpy arrays have the same dtype
            elif received[key].dtype != expected[key].dtype:
                explanation = (
                    'FAILED: data types do not match:\n' +
                    'Expected type: {}'.format(expected[key].dtype) + '\n' +
                    'Received type: {}'.format(received[key].dtype)
                )
                report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, explanation))
                fail = True

            else:
                expected[key] = np.atleast_1d(expected[key])
                received[key] = np.atleast_1d(received[key])

                # Checking if the arrays have the same number of dimensions
                if len(expected[key].shape) != len(received[key].shape):
                    explanation = (
                        'FAILED: tensor dimensionalities do not match:\n' +
                        'Expected shape: {}\n'.format(expected[key].shape) +
                        'Received shape: {}'.format(received[key].shape)
                    )
                    report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, explanation))
                    fail = True

                # Checking if the arrays have the same shape
                elif expected[key].shape != received[key].shape:
                    explanation = (
                        'FAILED: tensor shapes do not match:\n' +
                        'Expected shape: {}\n'.format(expected[key].shape) +
                        'Received shape: {}'.format(received[key].shape)
                    )
                    report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, explanation))
                    fail = True

                # Checking the arrays that have the exact types
                elif received[key].dtype in [numpy.bool_, numpy.byte, numpy.ubyte]:
                    if not (received[key] == expected[key]).all():
                        explanation = (
                            'FAILED: values do not match:\n' +
                            'Expected value: {}\n'.format(expected[key]) +
                            'Received value: {}'.format(received[key])
                        )
                        report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, explanation))
                        fail = True

                # Checking arrays with non-numeric dtypes (strings, bytes, objects) via exact equality
                elif not numpy.issubdtype(received[key].dtype, numpy.number):
                    if not (received[key] == expected[key]).all():
                        explanation = (
                            'FAILED: values do not match:\n' +
                            'Expected value: {}\n'.format(expected[key]) +
                            'Received value: {}'.format(received[key])
                        )
                        report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, explanation))
                        fail = True

                else:
                    # Checking that the arrays do not contain NaNs (they should not)
                    if not numpy.isfinite(received[key]).all() or not numpy.isfinite(expected[key]).all():
                        report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, 'FAILED: answer contains NaN or Inf values'))
                        fail = True

                    # Checking the array values
                    else:
                        mismatch = numpy.abs((received[key] - expected[key]))
                        max_mismatch = mismatch.max()
                        if not max_mismatch < eps:
                            explanation = (
                                'FAILED: answers do not match:\n' +
                                'Maximal allowed mismatch: {}\n'.format(str(eps)) +
                                'Received mismatch: {}'.format(str(max_mismatch))
                            )
                            report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, False, explanation))
                            fail = True

        if not fail:
            if not report_fails_only:
                report.append(_build_status(test_info + ANSWER_KEY_SEPARATOR + key, True, 'OK'))

    return report


def print_report(report: typing.List[typing.Dict[str, typing.Any]]) -> None:
    """
    Pretty-prints a run_checks() report for inline notebook feedback.

    Only the failing items are shown, each with its name and explanation.
    If there are no failures (an empty report, or one made only of passing
    entries), a single 'OK' line is printed instead.

    :param report: list of status dicts, as returned by run_checks()
    """
    failures = [entry for entry in report if not entry['ok']]

    if not failures:
        print('OK')
        return

    for entry in failures:
        print(f"FAILED: {entry['name']}")
        if 'status' in entry:
            print(entry['status'])


_PROGRESS_SHOWN_FAILURES = 3
_PROGRESS_LINE_WIDTH = 100


def _tensors_to_numpy(value: typing.Any) -> typing.Any:
    """Detach every torch tensor nested in ``value`` and turn it into numpy."""
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _tensors_to_numpy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tensors_to_numpy(item) for item in value)
    return value


def _format_value(value: np.ndarray | None) -> str:
    """Show a value in at most ~40 characters: scalars as-is, arrays abridged."""
    if value is None:
        return 'None'
    if value.size == 1:
        item = value.reshape(-1)[0].item()
        text = f'{item:.4g}' if isinstance(item, float) else repr(item)
    else:
        text = np.array2string(value, threshold=6, edgeitems=2, precision=4,
                               separator=', ')
    text = ' '.join(text.split())
    return text if len(text) <= 40 else text[:39] + '…'


def _describe_mismatch(expected: np.ndarray | None,
                       received: np.ndarray | None) -> str:
    """Say in one short phrase how ``received`` differs from ``expected``."""
    if expected is None or received is None:
        return f'expected {_format_value(expected)}, got {_format_value(received)}'
    if expected.dtype.kind in 'USOb' or received.dtype.kind in 'USOb':
        return f'expected {_format_value(expected)}, got {_format_value(received)}'
    if np.atleast_1d(expected).shape != np.atleast_1d(received).shape:
        return f'shape {received.shape} instead of {expected.shape}'
    if expected.dtype != received.dtype:
        return f'type {received.dtype} instead of {expected.dtype}'
    if not (np.isfinite(expected).all() and np.isfinite(received).all()):
        return 'contains NaN or Inf'
    if expected.size == 0:
        return f'expected an empty array, got {received.size} values'
    mismatch = float(np.abs(received.astype(float) - expected.astype(float)).max())
    if expected.size == 1:
        shown_expected, shown_received = _format_value(expected), _format_value(received)
        if shown_expected == shown_received:  # differs beyond the shown digits
            return f'expected {shown_expected}, got {shown_received} (off by {mismatch:.2g})'
        return f'expected {shown_expected}, got {shown_received}'
    return f'values off by up to {mismatch:.3g}'


def print_progress(
        title: str,
        answer: typing.Dict[str, typing.Any],
        prefixes: typing.Sequence[str],
        is_small_run: bool,
        answer_path: str | Path = 'yanswer.npz') -> None:
    """
    Print how one notebook step compares with the public answer key.

    Only the answer keys whose names start with one of ``prefixes`` are
    compared, with the same rules as the grader (``run_checks``). Torch
    tensors in ``answer`` are detached and converted to numpy first. The
    public key matches the small run only, so a full run prints a note
    instead of a comparison.

    Parameters
    ----------
    title : str
        Step name printed as the heading, e.g. ``'Task 1: Load the Data'``.
    answer : dict
        The notebook's answer so far; it is not modified.
    prefixes : sequence of str
        Starts of the flattened answer keys that belong to this step,
        e.g. ``('X_train', 'X_valid')``.
    is_small_run : bool
        Whether the notebook runs the small configuration the public
        answer key was made with.
    answer_path : str or Path
        Public answer key next to the notebook.

    Returns
    -------
    None
        The result is printed: a pass line, or the number of failing
        checks and the first three of them, one line each.

    If the public answer has no key for this step yet (the file is older
    than the notebook), a note is printed instead of a comparison; the
    next grading run writes a fresh file.

    Examples
    --------
    With ``yanswer.npz`` holding ``{'seed_check': [0.84, 0.76]}``:

    >>> print_progress('Task 0: Seeds', {'seed_check': [0.84, 0.76]},
    ...                prefixes=('seed_check',), is_small_run=True)
    Task 0: Seeds
      ✓ public checks pass (2 of 2)
    """
    print(title)
    if not is_small_run:
        print('  – full run: public answers match the small run only'
              ' (IS_UNITTEST = True)')
        return
    if not Path(answer_path).is_file():
        print(f'  – no public answers to compare with ({Path(answer_path).name} is missing)')
        return

    prefixes = tuple(prefixes)
    expected = {key: value for key, value in load_answer(answer_path).items()
                if key.startswith(prefixes)}
    if not expected:
        print(f'  – no public answers for this step in {Path(answer_path).name} yet;'
              ' the next check of the task refreshes it')
        return

    test_info = 'public'
    received = normalize_answer(_tensors_to_numpy(answer))
    report = run_checks(received, expected, test_info=test_info,
                        report_fails_only=False, include_explanations=False)
    failures = [entry for entry in report if not entry['ok']]
    if not failures:
        print(f'  ✓ public checks pass ({len(report)} of {len(report)})')
        return

    # run_checks' default tolerance, which the grader uses too.
    print(f'  ✗ public checks fail ({len(failures)} of {len(report)};'
          ' numbers must match within 1e-4)')
    for entry in failures[:_PROGRESS_SHOWN_FAILURES]:
        key = entry['name'].removeprefix(test_info + ANSWER_KEY_SEPARATOR)
        if key in received:
            reason = _describe_mismatch(expected[key], received[key])
        else:
            reason = 'missing from your answer'
        line = f'    {key}: {reason}'
        if len(line) > _PROGRESS_LINE_WIDTH:
            line = line[:_PROGRESS_LINE_WIDTH - 1] + '…'
        print(line)
    hidden_count = len(failures) - _PROGRESS_SHOWN_FAILURES
    if hidden_count > 0:
        print(f'    … and {hidden_count} more')


def save_answer(answer: typing.Dict[str, typing.Any],
                filename: str | Path | None = None) -> None:
    """
    Save a notebook's answer for grading, naming every value left out.

    Torch tensors are detached and converted to numpy, then the answer is
    flattened like ``normalize_answer`` does. An ``.npz`` file cannot hold
    ``None`` or arbitrary Python objects, so such values are not saved;
    their keys are printed instead of being dropped silently.

    Parameters
    ----------
    answer : dict
        The notebook's answer; it is not modified.
    filename : str or Path, optional
        Where to save. Defaults to ``answer.npz`` in ``get_data_dir()``,
        the file the grader reads.

    Returns
    -------
    None

    Examples
    --------
    >>> save_answer({'loss': 0.25, 'optimizer_defaults': {'fused': None}})
    Not saved (an .npz file cannot hold these values): optimizer_defaults -> fused
    """
    flat_answer = normalize_answer(_tensors_to_numpy(answer))
    savable = {key: value for key, value in flat_answer.items()
               if value is not None and value.dtype != object}
    left_out = [key for key in flat_answer if key not in savable]
    if left_out:
        print('Not saved (an .npz file cannot hold these values): '
              + ', '.join(left_out))
    if filename is None:
        filename = get_data_dir() / 'answer.npz'
    save_npz(savable, filename)


def get_teacher_path(task_path: str, fname: str = 'answer_benchmark.npz'):
    return Path(task_path) / fname


def get_student_path(task_path: str, fname: str='answer.npz'):
    return Path(task_path) / fname


def get_report_path(task_path: str, fname: str='.report.json'):
    return Path(task_path) / fname


def _load_npz(path: str):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def is_valid_npz(path: str):
    """
    Checks that the npz file exists and can be np.loaded.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            list(data.files)
        return True
    except Exception:
        return False


def run_script(notebook_path:str = '.task.ipynb'):

    if notebook_path.split('.')[-1] == 'ipynb':
        nb_executor.execute_notebook(Path(notebook_path))
    elif notebook_path.split('.')[-1] == 'py':
        spec = importlib.util.spec_from_file_location(Path(notebook_path).stem,
                                                      notebook_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.app.run()


def _replace_with_copy(source: Path, destination: Path):
    """Atomically replace a file with a copy of another file."""
    # Keep the temporary file beside the destination so that replace() stays
    # on one filesystem and cannot expose a partially copied inputs file.
    with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        shutil.copy2(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


_fallback_data_dir: typing.Optional[Path] = None


def get_data_dir() -> Path:
    """
    Return the checker's hidden working directory for the current run.

    check_code_task() sets this for the duration of a grading run, and it
    is deleted (along with everything in it) the moment that run ends --
    nothing here is meant to persist or be student-visible. Outside that
    context (e.g. a student interactively running task.py on its own,
    never through check_code_task()), falls back to a private, per-process
    temporary directory rather than raising, so a write that only matters
    during grading doesn't crash an otherwise-working interactive session.

    :return: Path to the current hidden data directory
    """
    value = os.environ.get('CHECKER_DATA_DIR')
    if value:
        return Path(value)

    global _fallback_data_dir
    if _fallback_data_dir is None:
        _fallback_data_dir = Path(tempfile.mkdtemp(prefix='checker_data_'))
    return _fallback_data_dir


def is_unittest_run() -> bool:
    """
    Return whether the code runs inside a grading run (``check_code_task()``).

    Notebooks use this to force the small, fast configuration during
    grading. ``check_code_task()`` sets ``CHECKER_DATA_DIR`` for the
    duration of the run and removes it afterwards, so an interactive
    session never sees it.

    Returns
    -------
    bool
        ``True`` while ``check_code_task()`` is running, ``False`` otherwise.

    Examples
    --------
    >>> is_unittest_run()
    False
    """
    return 'CHECKER_DATA_DIR' in os.environ


def _restore_or_delete(backup: Path, target: Path):
    """Replace `target` with `backup`'s contents if `backup` exists, else
    remove `target` entirely.

    Used to roll a student-visible file back to exactly its pre-run state
    after a crash, rather than guessing whether whatever it currently
    holds is safe to keep.
    """
    if backup.exists():
        _replace_with_copy(backup, target)
    else:
        target.unlink(missing_ok=True)


def refresh_preserved_inputs(inputs_path: str | Path):
    """Make the end-of-run restore bring back the current `inputs_path`.

    check_code_task() puts the caller's original `xinputs.npz` back when a run
    ends. A task that regenerates its inputs because the shipped file is stale
    (for example when card texts were edited) calls this right after writing
    the new file; otherwise the stale original would overwrite it and the
    problem would repeat on every run. Does nothing outside a grading run.
    """
    preserved = get_data_dir() / '_original_xinputs.npz'
    if preserved.exists():
        _replace_with_copy(Path(inputs_path), preserved)


def check_code_task():
    """Run the code-task checker and restore student-visible input data."""
    # Unlike the checker's own hidden working files, these two live at the
    # task root -- they're what a task's own notebook loads directly, by a
    # plain relative name, both during grading and interactively.
    active_inputs = Path('xinputs.npz')
    active_benchmark = Path('yanswer.npz')

    with tempfile.TemporaryDirectory() as data_dir_name:
        data_dir = Path(data_dir_name)
        os.environ['CHECKER_DATA_DIR'] = str(data_dir)

        original_inputs = data_dir / '_original_xinputs.npz'
        if active_inputs.exists():
            # Preserve the caller's student-visible inputs across both grading
            # stages, including failures while executing either notebook.
            shutil.copy2(active_inputs, original_inputs)

        original_benchmark = data_dir / '_original_yanswer.npz'
        if active_benchmark.exists():
            shutil.copy2(active_benchmark, original_benchmark)

        try:
            result = _check_code_task(
                data_dir=data_dir, active_inputs=active_inputs, active_benchmark=active_benchmark,
            )
        except BaseException:
            # A crash partway through a run must never leave partially
            # generated or misattributed data behind -- roll both files
            # back to exactly what existed before this run started.
            _restore_or_delete(original_inputs, active_inputs)
            _restore_or_delete(original_benchmark, active_benchmark)
            raise
        else:
            public_inputs = data_dir / 'inputs_public.npz'
            inputs_restored = False

            if public_inputs.exists():
                # The teacher notebook has just generated the public inputs
                # from scratch, so they are the current ones. Putting the
                # shipped copy back instead would keep a stale file alive
                # forever: every later run would regenerate, mismatch and
                # throw the student's saved progress away again.
                _replace_with_copy(public_inputs, active_inputs)
                inputs_restored = True
            elif original_inputs.exists():
                _replace_with_copy(original_inputs, active_inputs)
                inputs_restored = True
            elif active_inputs.exists():
                # No public stage exists for this task at all (e.g. cards
                # tasks, which only ever generate inputs_private.npz).
                # What's currently in active_inputs is the task's only
                # input, safely generated by a run that completed without
                # error.
                inputs_restored = True

            # A missing inputs file is safer than exposed private inputs.
            if not inputs_restored:
                active_inputs.unlink(missing_ok=True)

            return result
        finally:
            os.environ.pop('CHECKER_DATA_DIR', None)
            # Everything else -- answer.npz, .answer.npz, inputs_public.npz,
            # inputs_private.npz, the original-file backups above -- lives
            # only inside data_dir, which this `with` block deletes on exit.


def _check_code_task(data_dir: Path, active_inputs: Path, active_benchmark: Path):
    print("Checking code task", flush=True, file=sys.stderr)

    if os.path.exists('.task.ipynb'):
        # Ipynb task case
        student_script = 'task.ipynb'
        teacher_script = '.task.ipynb'
    elif os.path.exists('.task.py'):
        # Python task case (including Marimo)
        student_script = 'task.py'
        teacher_script = '.task.py'
    else:
        # Teacher did not provide a correct script file to run
        raise FileNotFoundError(
            (f"Could not find a python script (task.py) or a notebook (task.ipynb) "
             "to run. Please contact the author of the course.")
        )

    # Running benchmark notebook to generate all the files
    # takes inputs from xinputs.npz
    run_script(notebook_path=str(teacher_script))
    # generates inputs_public.npz, inputs_private.npz, xinputs.npz
    # puts outputs into answer.npz

    # If public inputs are present -- making a public check first
    if os.path.exists(data_dir / 'inputs_public.npz'):
        print("Running public tests", flush=True, file=sys.stderr)

        # Making the public benchmark answer available to the student
        shutil.copy(data_dir / 'answer.npz', active_benchmark)

        # Inputs are now public inputs
        shutil.copy(data_dir / 'inputs_public.npz', active_inputs)

        run_script(notebook_path=teacher_script)

        # Moving the answer to the hidden file
        shutil.copy(data_dir / 'answer.npz', data_dir / '.answer.npz')

        # Running student's notebook
        run_script(notebook_path=student_script)

        # Checking the public answer
        public_status = run_checks(
            load_answer(data_dir / 'answer.npz'),
            load_answer(data_dir / '.answer.npz'),
            test_info='public',
            report_fails_only=False,
            include_explanations=True
        )

        # Removing the answers
        (data_dir / 'answer.npz').unlink(missing_ok=True)
        print("Done", flush=True, file=sys.stderr)

        public_checks_passed = all(entry['ok'] for entry in public_status)
        if not public_checks_passed:
            # Do not run private inputs after public failure: their result must
            # reveal only pass/fail once the public stage has succeeded.
            return public_status
    else:
        public_status = []

    # Copying private inputs
    if os.path.exists(data_dir / 'inputs_private.npz'):
        shutil.copy(data_dir / 'inputs_private.npz', active_inputs)
    else:
        raise FileNotFoundError("Private inputs are missing. Contact the course author to fix it.")

    print("Running private tests", flush=True, file=sys.stderr)
    # Running teacher's notebook
    run_script(
        notebook_path=teacher_script
    )

    # Moving the answer to the hidden file
    shutil.copy(data_dir / 'answer.npz', data_dir / '.answer.npz')

    # Running student's notebook
    run_script(
        notebook_path=student_script
    )

    # Checking the public answer
    private_status = run_checks(
        load_answer(str(data_dir / 'answer.npz')),
        load_answer(str(data_dir / '.answer.npz')),
        test_info='private',
        report_fails_only=False,
        include_explanations=False
    )
    print("Done", flush=True, file=sys.stderr)

    status = public_status + private_status

    return status


def make_test(name: str, entry: typing.Dict[str, typing.Any]):
    def test(self):
        if entry['ok']:
            return
        status = f"\nItem: {name}"
        if 'status' in entry:
            status = (
                f"\nItem: {name}\n"
                f"Message:\n"
                f"===\n"
                f"{entry['status']}\n"
                f"===")
        # Raised directly (not via assertTrue) so the exception's own
        # message is exactly `status`, with none of assertTrue's
        # "<expr> is not true : " boilerplate prepended.
        raise self.failureException(status)
    return test


def build_tests_from_status(TestCase, status):
    width = len(str(len(status)))
    for idx, entry in enumerate(status):
        name = entry['name']
        name = f"test {idx:0{width}d} {name}"
        setattr(TestCase, name, make_test(name, entry))
