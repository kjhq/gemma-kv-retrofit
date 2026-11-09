#!/usr/bin/env bash
set -euo pipefail
R=$HOME/.cache/gemma-kv-retrofit/data/raw
hf download --repo-type dataset HuggingFaceFW/fineweb-edu sample/10BT/013_00000.parquet sample/10BT/000_00000.parquet --local-dir $R/fineweb-edu
hf download --repo-type dataset emozilla/pg19 data/test-00000-of-00001-29a571947c0b5ccc.parquet data/validation-00000-of-00001-0f92e2337f79aeac.parquet --local-dir $R/pg19
hf download --repo-type dataset codeparrot/codeparrot-clean-valid --local-dir $R/codeparrot-valid
touch $R/DONE
