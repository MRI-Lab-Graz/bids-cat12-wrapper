#!/bin/bash
# Quick test of config reading logic

cd "$(dirname "$0")"

CONFIG_JSON="config/config.json"
MODALITY="vbm"
SESSIONS=""
COVARIATES=""

echo "Testing config reading..."
echo "CONFIG_JSON = $CONFIG_JSON"
echo "MODALITY = $MODALITY"
echo ""

# Test sessions reading
if [[ -z "$SESSIONS" ]]; then
    SESSIONS=$(python3 << PYEOF
import json
try:
    with open("$CONFIG_JSON") as f:
        config = json.load(f)
    sessions = config.get("analysis", {}).get("sessions", ["all"])
    if sessions == ["all"]:
        print("all")
    else:
        print(",".join(str(s) for s in sessions))
except Exception as e:
    print(f"ERROR: {e}")
    print("all")
PYEOF
)
    echo "Sessions from config = '$SESSIONS'"
fi

# Test covariates reading
if [[ -z "$COVARIATES" ]]; then
    COVARIATES=$(python3 << PYEOF
import json
try:
    with open("$CONFIG_JSON") as f:
        config = json.load(f)
    modality_name = "$MODALITY"
    for mod in config.get("analysis", {}).get("modalities", []):
        if mod.get("name") == modality_name:
            covs = mod.get("covariates", [])
            if covs:
                print(",".join(covs))
            break
except Exception as e:
    print(f"ERROR: {e}")
PYEOF
)
    echo "Covariates from config = '$COVARIATES'"
fi

echo ""
echo "Expected: Sessions = '1,2'"
echo "Expected: Covariates = 'tiv,sex,age'"
