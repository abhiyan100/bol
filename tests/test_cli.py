"""CLI surface: exit codes, the doctor report, and clean startup errors."""

import errno
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
