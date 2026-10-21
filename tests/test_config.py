import os
import subprocess
import sys

import pytest


def run_python(code, **settings):
    return subprocess.run([sys.executable, '-c', code],
                          env={**os.environ, 'PYTHON_DOTENV_DISABLED': '1', **settings},
                          capture_output=True, text=True, timeout=20)


def test_cold_import_without_keys_or_model():
    result = run_python('''
import sys
from app.main import health as root
assert root()['gemini_configured'] is False
assert root()['api_key_configured'] is False
assert 'sentence_transformers' not in sys.modules
''', GEMINI_API_KEY='', API_KEY='', HF_HUB_OFFLINE='1')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('settings', [
    {'CHUNK_SIZE': '0'}, {'CHUNK_SIZE': '10', 'CHUNK_OVERLAP': '10'},
    {'CHUNK_OVERLAP': '-1'}, {'TOP_K': '0'}, {'MAX_UPLOAD_SIZE_MB': '0'},
    {'SIMILARITY_THRESHOLD': '-1'}, {'SIMILARITY_THRESHOLD': 'nan'}, {'SIMILARITY_THRESHOLD': 'inf'},
])
def test_invalid_config_fails_early(settings):
    result = run_python('import app.config', **settings)
    assert result.returncode != 0 and 'ValueError' in result.stderr
