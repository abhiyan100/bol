"""CLI surface: exit codes, the doctor report, and clean startup errors."""

import errno
import json
import os
import subprocess
import sys
import types

import pytest

import bol.cli as cli
from bol.config import Config

PROBES = (
    "probe_system",
    "probe_packages",
    "probe_wiring",
    "probe_permissions",
    "probe_weights",
    # Wake is on by default, so this one reports on a keyword model and a
    # package the machine running the tests may or may not have. Stubbed like
    # the rest, or "all green" would mean "green on this laptop".
    "probe_wake",
    "probe_injection",
)


@pytest.fixture
def green_doctor(monkeypatch):
    """Every probe passes, so nothing here touches the real machine."""
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: Config())
    for name in PROBES:
        monkeypatch.setattr(
            cli, name, lambda _cfg, _n=name: [(cli.OK, _n.removeprefix("probe_"), "")]
        )
    return monkeypatch


@pytest.fixture
def fake_daemon(monkeypatch):
    """Install a bol.daemon stub whose run() raises whatever the test wants."""

    def install(exc):
        module = types.ModuleType("bol.daemon")

        class Daemon:
            def __init__(self, cfg, text_mode=False):
                self.cfg = cfg

            async def run(self):
                raise exc

        module.Daemon = Daemon
        monkeypatch.setitem(sys.modules, "bol.daemon", module)

    monkeypatch.setattr(cli, "load_config", lambda *a, **k: Config())
    monkeypatch.setattr(cli, "_url", lambda cfg: "http://127.0.0.1:8770/hook?token=t")
    monkeypatch.setattr(cli.installer, "installed", lambda *a, **k: True)
    monkeypatch.setattr(cli, "_warn_missing_weights", lambda cfg: None)
    return install


def test_version_exits_zero(capsys):
    assert cli.main(["--version"]) == 0
    assert "bol" in capsys.readouterr().out


def test_help_exits_zero():
    assert cli.main(["--help"]) == 0


def test_unknown_flag_does_not_traceback():
    assert cli.main(["--nope"]) == 2


def test_doctor_all_green(green_doctor, capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "all good, run `bol run`" in out
    assert "[!! ]" not in out
    assert "—" not in out


def test_doctor_reports_a_failure(green_doctor, capsys):
    green_doctor.setattr(
        cli,
        "probe_permissions",
        lambda _cfg: [(cli.BAD, "input monitoring: not granted", "turn it on")],
    )
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "[!! ] input monitoring: not granted" in out
    assert "fix: turn it on" in out
    assert "fix the [!!] lines above" in out


def test_bare_bol_is_bol_run(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["text"] = args.text
        return 0

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    assert cli.main([]) == 0
    assert seen == {"text": False}


@pytest.mark.parametrize(
    "exc, expected",
    [
        (PermissionError("Input Monitoring is off, so the hotkey can't fire."),
         "Input Monitoring is off"),
        (ValueError("no input device matches 'webcam'."), "no input device matches"),
        (OSError(errno.EADDRINUSE, "Address already in use"), "is busy"),
    ],
)
def test_run_prints_one_clean_line(fake_daemon, capsys, exc, expected):
    fake_daemon(exc)
    assert cli.main(["run"]) == 1
    out = capsys.readouterr().out
    assert expected in out
    assert "Traceback" not in out
    assert len([line for line in out.splitlines() if line.strip()]) == 1


def test_run_reports_a_bridge_error(fake_daemon, capsys):
    from bol.bridge import BridgeError

    fake_daemon(BridgeError("couldn't reach Claude Code"))
    assert cli.main(["run"]) == 1
    assert "couldn't reach Claude Code" in capsys.readouterr().out


def test_run_says_bye_on_ctrl_c(fake_daemon, capsys):
    fake_daemon(KeyboardInterrupt())
    assert cli.main(["run"]) == 0
    assert "bye" in capsys.readouterr().out


def test_wanted_models_follows_the_config():
    cfg = Config()
    cfg.stt.engine = "none"
    cfg.llm.provider = "off"
    cfg.cleanup.mode = "off"
    cfg.tts.engine = "kokoro"
    assert cli.wanted_models(cfg) == [("voice", cfg.tts.kokoro_model)]


def test_quiet_model_libraries_goes_offline_when_everything_is_cached(monkeypatch):
    import bol.llm.engine as engine

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.setattr(engine, "hf_available", lambda: True)
    monkeypatch.setattr(engine, "weights_cached", lambda repo: True)
    monkeypatch.setattr(cli, "installed_models", lambda cfg: [("summaries", "x/y")])
    cli._quiet_model_libraries(Config())
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_quiet_model_libraries_stays_online_when_a_model_is_missing(monkeypatch):
    import bol.llm.engine as engine

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(engine, "hf_available", lambda: True)
    monkeypatch.setattr(engine, "weights_cached", lambda repo: repo != "missing/one")
    monkeypatch.setattr(
        cli, "installed_models", lambda cfg: [("summaries", "x/y"), ("cleanup", "missing/one")]
    )
    cli._quiet_model_libraries(Config())
    assert "HF_HUB_OFFLINE" not in os.environ


def test_quiet_model_libraries_respects_an_explicit_choice(monkeypatch):
    import bol.llm.engine as engine

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(engine, "hf_available", lambda: True)
    monkeypatch.setattr(engine, "weights_cached", lambda repo: True)
    monkeypatch.setattr(cli, "installed_models", lambda cfg: [("summaries", "x/y")])
    cli._quiet_model_libraries(Config())
    assert os.environ["HF_HUB_OFFLINE"] == "0"


# ------------------------------------------------------------------ bluetooth


class FakeStream:
    """sd.InputStream, as far as probe_microphone is concerned."""

    device = 3

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _frames):
        return None


def _fake_sounddevice(monkeypatch, name):
    module = types.ModuleType("sounddevice")
    module.InputStream = lambda **_kw: FakeStream()
    module.query_devices = lambda _device, _kind: {"name": name}
    monkeypatch.setitem(sys.modules, "sounddevice", module)


@pytest.fixture
def mic_probe(monkeypatch):
    """probe_microphone with the permission and device questions answered."""

    def arrange(name, bluetooth_json=None):
        _fake_sounddevice(monkeypatch, name)
        monkeypatch.setattr(cli, "_mic_authorization", lambda: cli._MIC_AUTHORIZED)
        monkeypatch.setattr(cli, "_input_device", lambda _cfg: None)
        monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            cli.subprocess,
            "run",
            lambda *a, **k: types.SimpleNamespace(
                returncode=0, stdout=bluetooth_json or "{}", stderr=""
            ),
        )
        return cli.probe_microphone(Config())

    return arrange


@pytest.mark.parametrize(
    "name",
    [
        "Abhiyan's AirPods Pro",
        "Beats Studio Buds",
        "Bluetooth Headset",
        "Sony WH-1000XM5",
        "BT Mic",
    ],
)
def test_a_bluetooth_mic_gets_a_row_of_its_own(mic_probe, name):
    rows = mic_probe(name)
    assert rows[0][0] == cli.OK
    assert name in rows[0][1]
    assert rows[1] == (cli.INFO, cli.BLUETOOTH_MIC_NOTE, "")
    assert "200 to 500 ms" in rows[1][1]


@pytest.mark.parametrize(
    "name",
    [
        "MacBook Pro Microphone",
        "Shure MV7",          # wired, and "MV7" must not read as a headset
        "Scarlett Solo USB",
    ],
)
def test_a_wired_mic_gets_no_extra_row(mic_probe, name):
    rows = mic_probe(name)
    assert len(rows) == 1
    assert rows[0] == (cli.OK, f"microphone: {name}", "")


def test_a_renamed_headset_is_caught_by_system_profiler(mic_probe):
    # People rename their headsets. The name says nothing, so the only
    # trustworthy answer is the list of connected Bluetooth devices.
    listing = json.dumps(
        {
            "SPBluetoothDataType": [
                {
                    "device_connected": [{"Kitchen Speaker": {"device_minorType": "Headset"}}],
                    "device_not_connected": [{"Old Mic": {}}],
                }
            ]
        }
    )
    rows = mic_probe("Kitchen Speaker", bluetooth_json=listing)
    assert rows[1][1] == cli.BLUETOOTH_MIC_NOTE


def test_a_disconnected_bluetooth_device_is_not_the_mic(mic_probe):
    listing = json.dumps(
        {"SPBluetoothDataType": [{"device_not_connected": [{"Old Mic": {}}]}]}
    )
    rows = mic_probe("Old Mic", bluetooth_json=listing)
    assert len(rows) == 1


def test_system_profiler_failing_is_not_a_doctor_failure(monkeypatch):
    # doctor may be less informative. It may not be slower than the person
    # reading it, and it may not fall over.
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")

    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired("system_profiler", 2.0)

    monkeypatch.setattr(cli.subprocess, "run", boom)
    assert cli.bluetooth_device_names() == []
    assert cli.is_bluetooth_mic("Kitchen Speaker") is False


def test_the_bluetooth_probe_is_bounded(monkeypatch):
    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs)
        seen["argv"] = args[0]
        return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.subprocess, "run", record)
    cli.bluetooth_device_names()
    assert seen["argv"] == ["system_profiler", "SPBluetoothDataType", "-json"]
    assert seen["timeout"] <= 2.0


def test_a_denied_microphone_still_reports_one_row(monkeypatch):
    monkeypatch.setattr(cli, "_mic_authorization", lambda: cli._MIC_DENIED)
    rows = cli.probe_microphone(Config())
    assert len(rows) == 1
    assert rows[0][0] == cli.BAD
