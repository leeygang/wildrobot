#!/bin/sh

# POSIX-compatible wrapper to convert URDF to USD via IsaacLab

PROJECT_ROOT="${HOME}/projects/wildrobot"
ISAACLAB_PATH="${HOME}/projects/IsaacLab"

INPUT_URDF="${PROJECT_ROOT}/assets/urdf/wildrobot.urdf"
OUTPUT_USD="${PROJECT_ROOT}/assets/usd/wildrobot_from_urdf.usd"

# Allow overrides via flags: --urdf <path> --out <path>
while [ $# -gt 0 ]; do
  case "$1" in
    --urdf)
      shift
      INPUT_URDF="$1"
      ;;
    --out)
      shift
      OUTPUT_USD="$1"
      ;;
    --verbose)
      VERBOSE="--verbose"
      ;;
    --headless)
      HEADLESS="--headless"
      ;;
    *)
      # Pass-through extra args to the converter
      EXTRA_ARGS="${EXTRA_ARGS} $1"
      ;;
  esac
  shift
done

echo "Converting URDF to USD..."
echo "Project root: ${PROJECT_ROOT}"
echo "IsaacLab path: ${ISAACLAB_PATH}"
echo "URDF input: ${INPUT_URDF}"
echo "USD output: ${OUTPUT_USD}"

# Basic checks
if [ ! -d "${ISAACLAB_PATH}" ]; then
  echo "Error: IsaacLab not found at ${ISAACLAB_PATH}" >&2
  exit 1
fi
if [ ! -f "${ISAACLAB_PATH}/isaaclab.sh" ]; then
  echo "Error: isaaclab.sh not found at ${ISAACLAB_PATH}/isaaclab.sh" >&2
  exit 1
fi
if [ ! -f "${INPUT_URDF}" ]; then
  echo "Error: URDF file not found at ${INPUT_URDF}" >&2
  exit 1
fi

# Run IsaacLab converter. Common location for the tool:
# scripts/tools/convert_urdf.py (IsaacLab provides this utility)
# Resolve absolute paths BEFORE changing working directory context
# Prefer readlink -f (Linux). Fallback to prefixing $PWD if path is relative.
resolve_abs() {
  _p="$1"
  case "$_p" in
    /*) echo "$_p" ;;
    *)
      if command -v readlink >/dev/null 2>&1; then
        readlink -f "$_p" 2>/dev/null || echo "$PWD/$_p"
      else
        echo "$PWD/$_p"
      fi
      ;;
  esac
}

ABS_URDF="$(resolve_abs "${INPUT_URDF}")"
ABS_OUT="$(resolve_abs "${OUTPUT_USD}")"

cd "${ISAACLAB_PATH}" || exit 1

./isaaclab.sh -p scripts/tools/convert_urdf.py \
  "${ABS_URDF}" \
  "${ABS_OUT}" \
  ${VERBOSE} ${HEADLESS} ${EXTRA_ARGS}

# Verify USD file was created successfully
if [ -f "${ABS_OUT}" ]; then
  # stat -c works on Linux; fallback to macOS -f
  USD_SIZE=$(stat -c%s "${ABS_OUT}" 2>/dev/null || stat -f%z "${ABS_OUT}" 2>/dev/null)
  if [ "${USD_SIZE}" -gt 1000 ] 2>/dev/null; then
    HR_SIZE=$(numfmt --to=iec-i --suffix=B "${USD_SIZE}" 2>/dev/null || echo "${USD_SIZE} bytes")
    echo "✓ Conversion complete! USD saved to ${ABS_OUT}"
    echo "  File size: ${HR_SIZE}"
    exit 0
  else
    echo "✗ Error: USD file created but appears to be empty or corrupted (size: ${USD_SIZE} bytes)" >&2
    exit 1
  fi
else
  echo "✗ Error: USD file was not created at ${ABS_OUT}" >&2
  exit 1
fi
