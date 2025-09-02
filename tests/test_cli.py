import subprocess, sys, os, json

def run_cli(args):
    return subprocess.run([sys.executable, "-m", "chunkviz.cli.app"] + args, capture_output=True, text=True)

def test_cli_ingest_and_chunk(tmp_path):
    # Make sample file
    f = tmp_path / "sample.txt"
    f.write_text("Hello world. This is a test file.")
    # Ingest
    r = run_cli(["ingest", str(f), "--out", str(tmp_path)])
    assert r.returncode == 0
    docs = json.load(open(tmp_path/"docs.json"))
    assert docs

    # Chunk
    r = run_cli(["chunk", "--config", "data/configs/demo_layout.yaml", "--run-name", "pytest_run"])
    assert r.returncode == 0
