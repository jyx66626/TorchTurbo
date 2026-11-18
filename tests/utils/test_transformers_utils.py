"""Regression tests for dynamic Transformers module imports."""

import importlib
import multiprocessing
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

dill = pytest.importorskip("dill")
transformers = pytest.importorskip("transformers")
from transformers import dynamic_module_utils

from torchturbo.utils import transformers_utils
from torchturbo.utils.transformers_utils import patch_dynamic_module_loading


@pytest.fixture(autouse=True)
def patched_loading(monkeypatch):
    # Restore the loader and installation state after each test.
    monkeypatch.setattr(
        dynamic_module_utils,
        "get_cached_module_file",
        dynamic_module_utils.get_cached_module_file,
    )
    monkeypatch.setattr(
        transformers_utils,
        "_patched_get_cached_module_file",
        transformers_utils._patched_get_cached_module_file,
    )
    monkeypatch.setattr(
        dynamic_module_utils,
        "get_class_in_module",
        dynamic_module_utils.get_class_in_module,
    )
    monkeypatch.setattr(
        transformers_utils,
        "_patched_get_class_in_module",
        transformers_utils._patched_get_class_in_module,
    )
    patch_dynamic_module_loading()


@pytest.fixture
def checkpoint(tmp_path):
    checkpoint = tmp_path / f"remote_code_{uuid4().hex}"
    checkpoint.mkdir()
    (checkpoint / "processing.py").write_text(
        "from typing import TypedDict\n\n"
        "class ProcessorKwargs(TypedDict):\n    size: int\n\n"
        "class Processor:\n    valid_kwargs = ProcessorKwargs\n"
    )
    yield checkpoint
    package = dynamic_module_utils.TRANSFORMERS_DYNAMIC_MODULE_NAME
    prefix = f"{package}.{checkpoint.name}"
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]
    parent = sys.modules.get(package)
    if parent is not None:
        vars(parent).pop(checkpoint.name, None)
    shutil.rmtree(
        Path(dynamic_module_utils.HF_MODULES_CACHE) / package / checkpoint.name,
        ignore_errors=True,
    )


def _load_processor(checkpoint):
    return dynamic_module_utils.get_class_from_dynamic_module(
        "processing.Processor", checkpoint, local_files_only=True
    )


def _unpickle_processor(payload, module_name):
    was_loaded = module_name in sys.modules
    processor = dill.loads(payload)
    module = importlib.import_module(module_name)
    return (
        was_loaded,
        type(processor) is module.Processor,
        processor.valid_kwargs is module.ProcessorKwargs,
    )


def test_dynamic_module_preserves_dill_identity(checkpoint):
    processor_class = _load_processor(checkpoint)
    assert dill.loads(dill.dumps(processor_class)) is processor_class
    assert type(dill.loads(dill.dumps(processor_class()))) is processor_class

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        result = pool.apply_async(
            _unpickle_processor,
            (dill.dumps(processor_class()), processor_class.__module__),
        )
        assert result.get(timeout=30) == (False, True, True)


def test_cached_and_forced_loads_update_parent_binding(checkpoint):
    processor_class = _load_processor(checkpoint)
    assert _load_processor(checkpoint) is processor_class
    module_name = processor_class.__module__
    parent_name, _, name = module_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    assert vars(parent)[name] is sys.modules[module_name]

    reloaded = dynamic_module_utils.get_class_in_module(
        "Processor", Path(*module_name.split(".")).with_suffix(".py"),
        force_reload=True,
    )
    assert reloaded is not processor_class
    assert vars(parent)[name] is sys.modules[module_name]
    assert dill.loads(dill.dumps(reloaded)) is reloaded


@pytest.mark.parametrize("preload_component", [False, True])
def test_reexported_class_binds_both_modules(checkpoint, preload_component):
    source = checkpoint / "processing.py"
    (checkpoint / "component.py").write_text(source.read_text())
    source.write_text("from .component import Processor\n")
    if preload_component:
        dynamic_module_utils.get_class_from_dynamic_module(
            "component.Processor", checkpoint, local_files_only=True
        )
    processor_class = _load_processor(checkpoint)
    parent_name, _, name = processor_class.__module__.rpartition(".")
    assert name == "component"
    parent = importlib.import_module(parent_name)
    assert vars(parent)["processing"] is sys.modules[f"{parent_name}.processing"]
    assert vars(parent)["component"] is sys.modules[f"{parent_name}.component"]
    assert dill.loads(dill.dumps(processor_class)) is processor_class


def test_repeated_patching_wraps_once(checkpoint):
    installed = dynamic_module_utils.get_class_in_module
    cached_loader = dynamic_module_utils.get_cached_module_file
    import_scanner = dynamic_module_utils.get_relative_import_files
    patch_dynamic_module_loading()
    patch_dynamic_module_loading()
    assert dynamic_module_utils.get_class_in_module is installed
    assert dynamic_module_utils.get_cached_module_file is cached_loader
    assert dynamic_module_utils.get_relative_import_files is import_scanner
    processor_class = _load_processor(checkpoint)
    assert dill.loads(dill.dumps(processor_class)) is processor_class


def test_repatching_preserves_outer_and_cached_wrappers(checkpoint, monkeypatch):
    installed = dynamic_module_utils.get_class_in_module
    outer = Mock(wraps=installed)

    with monkeypatch.context() as context:
        context.setattr(dynamic_module_utils, "get_class_in_module", outer)
        patch_dynamic_module_loading()
        repatched = dynamic_module_utils.get_class_in_module
        patch_dynamic_module_loading()
        assert dynamic_module_utils.get_class_in_module is repatched

        processor_class = _load_processor(checkpoint)
        outer.assert_called_once()
        assert dill.loads(dill.dumps(processor_class)) is processor_class

    assert dynamic_module_utils.get_class_in_module is installed
    assert _load_processor(checkpoint) is processor_class
    patch_dynamic_module_loading()
    assert _load_processor(checkpoint) is processor_class
    outer.assert_called_once()


@pytest.mark.parametrize("module_dir", ["", "components"])
@pytest.mark.parametrize("relative_path", [False, True])
def test_local_transitive_imports_and_cache_invalidation(
    checkpoint, monkeypatch, module_dir, relative_path
):
    source_dir = checkpoint / module_dir
    source_dir.mkdir(exist_ok=True)
    (source_dir / "processing.py").write_text(
        "from .helper import VALUE\nclass Processor:\n    value = VALUE\n"
    )
    (source_dir / "helper.py").write_text("from .base import VALUE\n")
    (source_dir / "base.py").write_text("VALUE = 1\n")
    model_path = checkpoint
    if relative_path:
        monkeypatch.chdir(checkpoint.parent)
        model_path = checkpoint.name
    module_file = str(Path(module_dir) / "processing.py")

    def load():
        cached = dynamic_module_utils.get_cached_module_file(
            model_path, module_file, local_files_only=True
        )
        processor = dynamic_module_utils.get_class_in_module("Processor", cached)
        return cached, processor

    first_cache, first_processor = load()
    assert first_processor.value == 1
    cache_dir = (Path(dynamic_module_utils.HF_MODULES_CACHE) / first_cache).parent
    assert (cache_dir / "base.py").read_text() == "VALUE = 1\n"
    assert load() == (first_cache, first_processor)
    assert dill.loads(dill.dumps(first_processor)) is first_processor

    # Only a transitive dependency changes; the entry point and helper do not.
    (source_dir / "base.py").write_text("VALUE = 2\n")
    second_cache, second_processor = load()
    assert second_cache != first_cache
    assert second_processor.value == 2
    assert first_processor.value == 1


def test_caching_circular_relative_imports_terminates(checkpoint, monkeypatch):
    entry = checkpoint / "processing.py"
    helper = checkpoint / "helper.py"
    entry.write_text("from .helper import VALUE\n")
    helper.write_text("from .processing import Processor\n")
    original = dynamic_module_utils.get_relative_imports
    calls = 0

    def bounded_scan(path):
        nonlocal calls
        calls += 1
        assert calls <= 10, "Circular imports were scanned repeatedly"
        return original(path)

    monkeypatch.setattr(dynamic_module_utils, "get_relative_imports", bounded_scan)
    cached = dynamic_module_utils.get_cached_module_file(checkpoint, "processing.py")
    cache_dir = (Path(dynamic_module_utils.HF_MODULES_CACHE) / cached).parent
    assert (cache_dir / "processing.py").read_text() == entry.read_text()
    assert (cache_dir / "helper.py").read_text() == helper.read_text()


@pytest.mark.parametrize(
    "version, should_patch",
    [
        ("5.5.4", True),
        ("5.9.0", True),
        ("5.10.0", True),
        ("5.10.1", False),
        ("5.11.0", False),
    ],
)
def test_transitive_import_patch_version_gate(monkeypatch, version, should_patch):
    original = Mock(return_value="cached/processing.py")
    scanner = Mock()
    monkeypatch.setattr(transformers, "__version__", version)
    monkeypatch.setattr(dynamic_module_utils, "get_cached_module_file", original)
    monkeypatch.setattr(dynamic_module_utils, "get_relative_import_files", scanner)
    patch_dynamic_module_loading()
    assert (dynamic_module_utils.get_cached_module_file is not original) == should_patch
    assert dynamic_module_utils.get_relative_import_files is scanner

    result = dynamic_module_utils.get_cached_module_file(
        "remote/model", "processing.py", "download_cache", revision="pinned", token="test"
    )
    assert result == "cached/processing.py"
    original.assert_called_once_with(
        "remote/model", "processing.py", "download_cache", revision="pinned", token="test"
    )
