from __future__ import annotations

import json
import sys
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from shared8_contract import (  # noqa: E402
    INPUT_SCHEMA,
    Shared8ContractError,
    assert_no_privileged_cli,
    load_input_manifest,
    source_aspect_bucket,
)
from ffprobe_pyav_compat import _input_path, _probe  # noqa: E402


def _rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": INPUT_SCHEMA,
            "index": index,
            "iid": f"{index:016x}",
            "split": "test" if index < 5 else "validation",
            "source_video": f"/tmp/shared8/source_{index}.mp4",
            "instruction": f"Do action {index}.",
            "seed": 2026 + index,
        }
        for index in range(8)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_closed_manifest_accepts_exact_source_instruction_rows(tmp_path: Path) -> None:
    path = tmp_path / "inputs.jsonl"
    _write_jsonl(path, _rows())
    _, loaded = load_input_manifest(path)
    assert [row.index for row in loaded] == list(range(8))
    assert loaded[7].iid == "0000000000000007"


@pytest.mark.parametrize(
    "privileged_key",
    ["target_video", "mask", "swept_tube", "track", "pose", "trajectory", "reference"],
)
def test_closed_manifest_rejects_privileged_fields(
    tmp_path: Path, privileged_key: str
) -> None:
    rows = _rows()
    rows[0][privileged_key] = "/forbidden"
    path = tmp_path / "inputs.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(Shared8ContractError, match="non-closed keys"):
        load_input_manifest(path)


def test_manifest_requires_exactly_eight_rows(tmp_path: Path) -> None:
    path = tmp_path / "inputs.jsonl"
    _write_jsonl(path, _rows()[:7])
    with pytest.raises(Shared8ContractError, match="exactly 8"):
        load_input_manifest(path)


def test_source_aspect_bucket_preserves_orientation_and_budget() -> None:
    height, width = source_aspect_bucket(height=1080, width=1920, max_pixels=832 * 480)
    assert width > height
    assert height % 16 == width % 16 == 0
    assert height * width <= 832 * 480


def test_cli_privileged_condition_guard() -> None:
    assert_no_privileged_cli(["python", "infer.py", "--source-video", "/source.mp4"])
    with pytest.raises(Shared8ContractError, match="privileged"):
        assert_no_privileged_cli(["python", "infer.py", "--target-video", "/target.mp4"])


@pytest.mark.parametrize(
    "launcher",
    [
        "auh_lucy_official_shared8.sbatch",
        "auh_bernini_full644_shared8.sbatch",
        "auh_omnivideo2_official_shared8.sbatch",
    ],
)
def test_sbatch_uses_exported_frozen_launcher_root(launcher: str) -> None:
    text = (MODULE_ROOT / "scripts" / launcher).read_text(encoding="utf-8")
    assert "ACTION_BASELINE_LAUNCHER_ROOT" in text
    assert 'dirname -- "${BASH_SOURCE[0]}"' not in text


def test_lucy_uses_one_slurm_gpu_per_array_sample() -> None:
    text = (MODULE_ROOT / "scripts" / "auh_lucy_official_shared8.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:mi210:1" in text
    assert "#SBATCH --array=0-7%8" in text
    assert "ROCR_VISIBLE_DEVICES" not in text
    assert "HIP_VISIBLE_DEVICES" not in text
    assert "CUDA_VISIBLE_DEVICES" not in text


def test_lucy_geometry_uses_stride32_token_safe_buckets() -> None:
    text = (MODULE_ROOT / "run_shared8.py").read_text(encoding="utf-8")
    assert "source_aspect_sqrt_max_pixels_then_floor_to_stride32" in text
    assert "required_pixel_stride" in text


def test_submitter_gates_lucy_rest_on_sample_zero_canary() -> None:
    text = (MODULE_ROOT / "scripts" / "submit_auh_shared8.sh").read_text(
        encoding="utf-8"
    )
    assert "lucy-canary --array=0" in text
    assert 'lucy-rest --array=1-7%8 --dependency="afterok:${canary_job_id}"' in text


def test_common_launcher_has_strict_pyav_probe_fallback() -> None:
    text = (MODULE_ROOT / "scripts" / "shared8_auh_common.sh").read_text(
        encoding="utf-8"
    )
    assert "ffprobe_pyav_compat.py" in text
    assert "source archive lacks an executable PyAV ffprobe backend" in text
    assert 'probe_backend="frozen-pyav"' in text
    assert "video_probe_runtime=" in text


def test_pyav_probe_accepts_only_shared8_contract_invocation() -> None:
    text = (MODULE_ROOT / "ffprobe_pyav_compat.py").read_text(encoding="utf-8")
    assert "stream=codec_name,width,height,avg_frame_rate,nb_read_frames" in text
    assert '"-count_frames"' in text
    assert "unsupported invocation" in text


def _fake_av(*, average_rate: object = Fraction(25, 1), duration: object = 3_240_000):
    stream = SimpleNamespace(
        index=1,
        average_rate=average_rate,
        base_rate=Fraction(30, 1),
        codec_context=SimpleNamespace(name="h264", width=640, height=360),
    )

    class FakeContainer:
        def __init__(self) -> None:
            self.streams = SimpleNamespace(video=[stream])
            self.duration = duration
            self.decoded_stream = None
            self.closed = False

        def decode(self, selected_stream):
            self.decoded_stream = selected_stream
            return iter(range(81))

        def close(self) -> None:
            self.closed = True

    container = FakeContainer()
    module = SimpleNamespace(open=lambda *_args, **_kwargs: container, time_base=1_000_000)
    return module, container, stream


def test_pyav_probe_decodes_v0_stream_object_for_audio_first_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, container, stream = _fake_av()
    monkeypatch.setitem(sys.modules, "av", module)
    result = _probe(tmp_path / "audio_first.mp4")
    assert container.decoded_stream is stream
    assert container.closed is True
    assert result["streams"][0]["nb_read_frames"] == "81"
    assert result["streams"][0]["avg_frame_rate"] == "25/1"


def test_pyav_probe_does_not_substitute_base_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, container, _stream = _fake_av(average_rate=None)
    monkeypatch.setitem(sys.modules, "av", module)
    with pytest.raises(SystemExit):
        _probe(tmp_path / "missing_average_rate.mp4")
    assert container.closed is True


def test_pyav_probe_does_not_synthesize_duration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, container, _stream = _fake_av(duration=None)
    monkeypatch.setitem(sys.modules, "av", module)
    with pytest.raises(SystemExit):
        _probe(tmp_path / "missing_duration.mp4")
    assert container.closed is True


def test_pyav_probe_rejects_non_contract_invocation(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not decoded by this test")
    with pytest.raises(SystemExit):
        _input_path(["-v", "error", str(video)])
