#!/bin/bash
# Test the analysis name + covariate suffix logic

ANALYSIS_NAME="vbm_9mm_3G_tiv_sex_age"
COV_SUFFIX="tiv_sex_age"

echo "ANALYSIS_NAME: $ANALYSIS_NAME"
echo "COV_SUFFIX: $COV_SUFFIX"
echo ""

if [[ "$ANALYSIS_NAME" != *"$COV_SUFFIX"* ]]; then
    echo "Covariate suffix NOT found in analysis name - would append"
    echo "Would create: ${ANALYSIS_NAME}_${COV_SUFFIX}"
else
    echo "Covariate suffix already in analysis name - SKIP append"
    echo "Keeping: $ANALYSIS_NAME"
fi

echo ""
echo "---"
echo ""

# Test 2: without covariates in name
ANALYSIS_NAME2="vbm_smooth_auto"
echo "ANALYSIS_NAME: $ANALYSIS_NAME2"
echo "COV_SUFFIX: $COV_SUFFIX"
echo ""

if [[ "$ANALYSIS_NAME2" != *"$COV_SUFFIX"* ]]; then
    echo "Covariate suffix NOT found in analysis name - would append"
    echo "Would create: ${ANALYSIS_NAME2}_${COV_SUFFIX}"
else
    echo "Covariate suffix already in analysis name - SKIP append"
fi
