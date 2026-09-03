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
    # The real config on this machine is nobody's business here, and its
    # hints would drown the one line these tests are about.
    monkeypatch.setattr(cli, "removed_keys", lambda *a, **k: [])
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
    cfg.talk_back = True
    cfg.stt.engine = "none"
    cfg.llm.provider = "off"
    cfg.cleanup.mode = "off"
    cfg.tts.engine = "kokoro"
    assert cli.wanted_models(cfg) == [("voice", cfg.tts.kokoro_model)]


# ------------------------------------------------------- one-way and two-way


def test_one_way_downloads_neither_the_summarizer_nor_the_voice():
    # The point of the default: `bol setup` on a 16 GB Mac fetches speech and
    # cleanup, and nothing that exists only to talk back.
    cfg = Config()
    cfg.tts.engine = "kokoro"

    roles = [role for role, _repo in cli.wanted_models(cfg)]

    assert roles == ["speech to text", "cleanup"]


def test_two_way_adds_the_summaries_model_and_the_voice():
    cfg = Config()
    cfg.talk_back = True
    cfg.tts.engine = "kokoro"

    models = cli.wanted_models(cfg)

    assert [role for role, _repo in models] == [
        "speech to text", "summaries", "cleanup", "voice",
    ]
    assert ("summaries", cfg.llm.local_model) in models


def test_setup_and_doctor_say_which_mode_the_config_selects():
    cfg = Config()
    one_way = cli.mode_line(cfg)
    assert "one-way" in one_way
    assert "--talk-back" in one_way

    cfg.talk_back = True
    two_way = cli.mode_line(cfg)
    assert "two-way" in two_way

    # And doctor's model section leads with it, so the list underneath is
    # read as "what this mode needs" and not "what Bol has".
    rows = cli.probe_weights(cfg)
    assert rows[0] == (cli.INFO, two_way, "")


def test_the_talk_back_flag_wins_over_the_config(fake_daemon, monkeypatch):
    seen = {}

    class Daemon:
        def __init__(self, cfg, text_mode=False):
            seen["talk_back"] = cfg.talk_back

        async def run(self):
            pass

    module = types.ModuleType("bol.daemon")
    module.Daemon = Daemon
    monkeypatch.setitem(sys.modules, "bol.daemon", module)
    # The file says one thing ...
    cfg = Config()
    cfg.talk_back = False
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)

    assert cli.main(["run", "--text", "--talk-back"]) == 0
    assert seen["talk_back"] is True  # ... the flag says the other

    cfg.talk_back = True
    assert cli.main(["run", "--text", "--no-talk-back"]) == 0
    assert seen["talk_back"] is False  # and it wins both ways


def test_without_the_flag_the_config_decides(fake_daemon, monkeypatch):
    seen = {}

    class Daemon:
        def __init__(self, cfg, text_mode=False):
            seen["talk_back"] = cfg.talk_back

        async def run(self):
            pass

    module = types.ModuleType("bol.daemon")
    module.Daemon = Daemon
    monkeypatch.setitem(sys.modules, "bol.daemon", module)
    cfg = Config()
    cfg.talk_back = True
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)

    assert cli.main(["run", "--text"]) == 0
    assert seen["talk_back"] is True


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


# ---------------------------------------------------------------- the wizard
#
# "default settings installation is not that good ... ask people questions
# while setting up so they can install talk-back if needed, the AI text
# cleanup if needed." Five questions at most, and every one of them decides
# a download or a settings file.

import io
import logging
import time
from pathlib import Path

from bol.config import load_config, render_default_config, write_config_values


@pytest.fixture
def answers(monkeypatch):
    """Feed the wizard a script, and give it a terminal to ask into."""

    def arrange(*replies, devices=("MacBook Pro Microphone",), on_path=("claude",)):
        queue = list(replies)
        monkeypatch.setattr(
            "builtins.input", lambda _prompt: queue.pop(0) if queue else ""
        )
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(cli, "input_device_names", lambda: list(devices))
        monkeypatch.setattr(cli, "_agents_on_path", lambda: list(on_path))

    return arrange


def test_the_wizard_writes_every_answer_into_the_config(answers, tmp_path):
    answers("3", "y", "n", "2", "2", devices=("Built-in", "Shure MV7"))
    cfg = Config()

    values = cli.run_wizard(cfg)

    assert values == {
        (None, "talk_back"): True,
        ("setup", "agents"): ["claude", "codex"],
        ("cleanup", "mode"): "off",
        ("cleanup", "model"): "",
        ("audio", "input_device"): "Shure MV7",
        ("tts", "engine"): "kokoro",
    }
    # And the same answers are on the Config the rest of setup reads.
    assert cfg.talk_back is True
    assert cfg.setup.agents == ["claude", "codex"]
    assert cfg.cleanup.mode == "off"
    assert cfg.audio.input_device == "Shure MV7"
    assert cfg.tts.engine == "kokoro"

    # ... and they survive a round trip through a file that did not exist.
    path, rewritten = write_config_values(values, tmp_path / "config.toml")
    assert rewritten is False
    written = load_config(path)
    assert written.talk_back is True
    assert written.cleanup.mode == "off"
    assert written.setup.agents == ["claude", "codex"]
    assert written.tts.engine == "kokoro"


def test_enter_five_times_is_a_good_install(answers, tmp_path):
    # Every default is what a fresh install wants: the agents on PATH,
    # dictation only, and the cleanup model on.
    answers("", "", "", "", "", on_path=("claude", "codex"))
    cfg = Config()

    values = cli.run_wizard(cfg)

    assert values[("setup", "agents")] == ["claude", "codex"]
    assert values[(None, "talk_back")] is False
    assert values[("cleanup", "mode")] == "always"
    assert values[("cleanup", "model")] == "abhiyan10/bol-cleanup-350m-4bit"
    # No talk-back, so no voice question; one microphone, so no mic question.
    assert ("tts", "engine") not in values
    assert ("audio", "input_device") not in values


def test_no_coding_agent_never_asks_about_talk_back(answers, capsys):
    answers("4", "", on_path=("claude", "codex"))
    cfg = Config()

    values = cli.run_wizard(cfg)

    assert values[("setup", "agents")] == []
    assert values[(None, "talk_back")] is False
    # The question was never put, so the cleanup answer is what "" answered.
    assert cli.TALK_BACK_Q not in capsys.readouterr().out


def test_the_default_agent_choice_follows_path(answers):
    for on_path, expected in (
        (("claude", "codex"), ["claude", "codex"]),
        (("claude",), ["claude"]),
        (("codex",), ["codex"]),
        ((), []),
    ):
        answers("", "", "", on_path=on_path)
        cfg = Config()
        assert cli.run_wizard(cfg)[("setup", "agents")] == expected


def test_a_rerun_defaults_to_what_the_config_already_says(answers):
    answers("", "", "", on_path=("claude", "codex"))
    cfg = Config()
    cfg.setup.agents = ["codex"]   # answered last time
    cfg.talk_back = True

    values = cli.run_wizard(cfg)

    assert values[("setup", "agents")] == ["codex"]
    assert values[(None, "talk_back")] is True  # Enter kept it on


def test_yes_takes_every_default_without_asking(monkeypatch, capsys):
    def boom(_prompt):
        raise AssertionError("--yes must not ask anything")

    monkeypatch.setattr("builtins.input", boom)
    monkeypatch.setattr(cli, "input_device_names", lambda: ["A", "B"])
    monkeypatch.setattr(cli, "_agents_on_path", lambda: ["claude"])
    cfg = Config()

    values = cli.run_wizard(cfg, assume_yes=True)

    assert values[("setup", "agents")] == ["claude"]
    assert values[(None, "talk_back")] is False
    assert values[("cleanup", "mode")] == "always"
    assert values[("audio", "input_device")] == "A"
    assert "no questions" in capsys.readouterr().out


def test_a_pipe_behaves_exactly_like_yes(monkeypatch):
    def boom(_prompt):
        raise AssertionError("a non-tty stdin must not be asked anything")

    monkeypatch.setattr("builtins.input", boom)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(cli, "input_device_names", lambda: [])
    monkeypatch.setattr(cli, "_agents_on_path", lambda: ["codex"])

    values = cli.run_wizard(Config())

    assert values[("setup", "agents")] == ["codex"]
    assert values[(None, "talk_back")] is False


def test_a_nonsense_answer_is_asked_again(answers, capsys):
    answers("9", "banana", "2", "", "")
    cfg = Config()

    values = cli.run_wizard(cfg)

    assert values[("setup", "agents")] == ["codex"]
    out = capsys.readouterr().out
    assert "please answer 1 to 4" in out


def test_an_existing_config_keeps_its_other_keys(answers, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[hotkey]\nkey = "f13"\n[vocabulary]\nwords = ["Abhiyan"]\n')
    answers("1", "y", "", "", "1")

    values = cli.run_wizard(Config())
    written, rewritten = write_config_values(values, path)

    assert rewritten is True  # tomli_w rewrote it, so the comments went
    cfg = load_config(written)
    assert cfg.hotkey.key == "f13"          # ... but nothing else did
    assert cfg.vocabulary.words == ["Abhiyan"]
    assert cfg.talk_back is True
    assert cfg.setup.agents == ["claude"]


def test_the_wizard_only_writes_keys_the_default_file_documents():
    # render_default_config raises on a key it cannot find, which is how a
    # wizard answer that silently went nowhere gets caught here instead of on
    # somebody's laptop.
    with pytest.raises(ValueError):
        render_default_config({("nowhere", "at_all"): 1})


# ------------------------------------------------------- what setup then does


@pytest.fixture
def setup_run(monkeypatch, tmp_path):
    """`bol setup` with the machine, the network and the models stubbed out."""
    import bol.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(cli, "removed_keys", lambda *a, **k: [])
    monkeypatch.setattr(cli, "_setup_wake", lambda cfg: True)
    monkeypatch.setattr(cli, "_url", lambda cfg: "http://127.0.0.1:8770/hook?token=t")
    for name in ("probe_permissions", "probe_wiring", "probe_wake", "probe_injection"):
        monkeypatch.setattr(cli, name, lambda _cfg: [])

    downloaded: list[tuple[str, str]] = []
    # wanted_models, not installed_models: this fixture is about the user's
    # choices, and the CI runner has no speech extras installed.
    monkeypatch.setattr(cli, "_setup_models", lambda cfg: downloaded.extend(
        cli.wanted_models(cfg)
    ) or True)
    wired: list[str] = []
    monkeypatch.setattr(
        cli.installer,
        "install",
        lambda url, scope="user", project_dir=None, agent="claude": wired.append(agent)
        or Path("/fake/settings.json"),
    )
    return types.SimpleNamespace(
        downloaded=downloaded, wired=wired, path=tmp_path / "config.toml"
    )


def test_setup_installs_hooks_only_for_the_chosen_agents(setup_run, answers, capsys):
    answers("2", "y", "", "", "1", on_path=("claude", "codex"))

    assert cli.main(["setup"]) == 0

    assert setup_run.wired == ["codex"]     # Claude Code was on PATH and skipped
    assert load_config(setup_run.path).setup.agents == ["codex"]


def test_setup_with_talk_back_off_touches_no_settings_file(setup_run, answers, capsys):
    answers("3", "n", "", "", on_path=("claude", "codex"))

    assert cli.main(["setup"]) == 0

    assert setup_run.wired == []
    out = capsys.readouterr().out
    assert cli.NO_HOOKS_LINE in out
    assert "bol hook install" in out


def test_setup_with_no_coding_agent_touches_no_settings_file(setup_run, answers):
    answers("4", "", on_path=("claude", "codex"))

    assert cli.main(["setup"]) == 0

    assert setup_run.wired == []
    assert load_config(setup_run.path).setup.agents == []


def test_setup_downloads_only_what_was_chosen(setup_run, answers, monkeypatch):
    # Dictation with cleanup on: speech and cleanup, and nothing that exists
    # only to talk back.
    answers("1", "n", "y", "", on_path=("claude",))
    assert cli.main(["setup"]) == 0
    assert [role for role, _repo in setup_run.downloaded] == [
        "speech to text", "cleanup"
    ]

    setup_run.downloaded.clear()
    setup_run.wired.clear()
    # Talk-back with Kokoro and no cleanup: the other three.
    answers("1", "y", "n", "2", on_path=("claude",))
    assert cli.main(["setup"]) == 0
    assert [role for role, _repo in setup_run.downloaded] == [
        "speech to text", "summaries", "voice"
    ]


def test_setup_yes_asks_nothing_and_still_writes_a_config(setup_run, monkeypatch):
    monkeypatch.setattr(cli, "input_device_names", lambda: [])
    monkeypatch.setattr(cli, "_agents_on_path", lambda: ["claude"])
    monkeypatch.setattr(
        "builtins.input", lambda _p: (_ for _ in ()).throw(AssertionError("asked"))
    )

    assert cli.main(["setup", "--yes"]) == 0

    cfg = load_config(setup_run.path)
    assert cfg.talk_back is False
    assert cfg.cleanup.mode == "always"
    assert setup_run.wired == []   # dictation, so nobody's settings were opened


# ------------------------------------------------------------ download output
#
# "the download output was a wall of Hugging Face progress bars and warnings."
# One line per model now, and it updates in place.


def test_the_progress_line_reads_like_a_download():
    line = cli.progress_line("speech to text", 1.14 * 1024**3, 2.3 * 1024**3, 48 * 1024**2)
    assert line == "  speech to text  1.14 GB / 2.30 GB  48 MB/s"


def test_a_download_prints_one_line_that_updates_in_place(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    def fake_snapshot_download(repo):
        assert repo == "abhiyan10/bol-cleanup-350m-4bit"
        for part in range(3):
            (cache / f"part{part}.bin").write_bytes(b"x" * 500_000)
            time.sleep(0.02)

    def size():
        return sum(p.stat().st_size for p in cache.iterdir())

    out = io.StringIO()
    ok = cli.download_one(
        "cleanup",
        "abhiyan10/bol-cleanup-350m-4bit",
        fake_snapshot_download,
        size,
        total=195 * 1024 * 1024,
        interval=0.01,
        out=out,
    )

    text = out.getvalue()
    assert ok is True
    # One line: everything before the final newline is carriage returns.
    assert text.count("\n") == 1
    frames = [frame.strip() for frame in text.split("\r") if frame.strip()]
    assert all(frame.startswith("cleanup") for frame in frames)
    assert any("/ 195 MB" in frame and "MB/s" in frame for frame in frames)
    assert frames[-1] == "cleanup  done  1 MB"


def test_a_failed_download_says_so_and_does_not_stop_setup(tmp_path):
    def boom(_repo):
        raise OSError("no route to host")

    out = io.StringIO()
    ok = cli.download_one("voice", "x/y", boom, lambda: 0, total=0, interval=0.01, out=out)

    assert ok is False
    assert "could not download x/y (no route to host)" in out.getvalue()
    assert "retry on first use" in out.getvalue()


def test_setup_silences_hugging_faces_own_reporting(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    for name in cli._DOWNLOAD_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)

    cli._quiet_downloads()

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    for name in cli._DOWNLOAD_LOGGERS:
        # ERROR, so the "unauthenticated requests" warning never prints and a
        # real failure still does.
        assert logging.getLogger(name).level == logging.ERROR


# ------------------------------------------------------------ hooks on demand


def test_run_never_installs_hooks_behind_your_back(fake_daemon, monkeypatch, capsys):
    installed = []
    monkeypatch.setattr(
        cli.installer, "install", lambda *a, **k: installed.append(a) or Path("/x")
    )
    monkeypatch.setattr(cli.installer, "installed", lambda *a, **k: False)
    fake_daemon(KeyboardInterrupt())

    assert cli.main(["run", "--text"]) == 0

    assert installed == []


def test_one_way_run_says_nothing_about_hooks(monkeypatch, capsys):
    cfg = Config()
    cfg.talk_back = False
    monkeypatch.setattr(cli.installer, "installed", lambda *a, **k: False)

    cli._warn_missing_hooks(cfg)

    assert capsys.readouterr().out == ""


def test_talk_back_without_hooks_prints_one_hint_line(monkeypatch, capsys):
    cfg = Config()
    cfg.talk_back = True
    cfg.setup.agents = ["claude", "codex"]
    monkeypatch.setattr(cli, "_url", lambda cfg: "http://127.0.0.1:8770/hook?token=t")
    monkeypatch.setattr(cli.installer, "installed", lambda url, agent="claude": False)

    cli._warn_missing_hooks(cfg)

    out = capsys.readouterr().out
    assert len([line for line in out.splitlines() if line.strip()]) == 1
    assert "talk-back needs hooks" in out
    assert "bol hook install" in out
    assert "--agent codex" in out


def test_talk_back_with_hooks_already_there_says_nothing(monkeypatch, capsys):
    cfg = Config()
    cfg.talk_back = True
    monkeypatch.setattr(cli, "_url", lambda cfg: "http://127.0.0.1:8770/hook?token=t")
    monkeypatch.setattr(cli.installer, "installed", lambda url, agent="claude": True)
    monkeypatch.setattr(cli, "_agents_on_path", lambda: ["claude"])

    cli._warn_missing_hooks(cfg)

    assert capsys.readouterr().out == ""


def test_doctor_does_not_fail_a_dictation_only_machine(monkeypatch):
    # No hooks anywhere, talk-back off: that is a working install, not a
    # broken one, and doctor must not send anyone off to fix it.
    cfg = Config()
    monkeypatch.setattr(cli, "_url", lambda cfg: "http://127.0.0.1:8770/hook?token=t")
    monkeypatch.setattr(cli.installer, "installed", lambda url, agent="claude": False)

    rows = cli.probe_wiring(cfg)

    assert not any(status == cli.BAD for status, _label, _hint in rows)
    assert rows[0] == (cli.INFO, "hooks: not needed for dictation (talk_back = false)", "")
