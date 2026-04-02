# assets/v2 (WildRobot full body with arms)

This folder is the **v2** robot asset variant (full body with arms).

- Onshape document (assembly): `WildRobot`
  - Link: https://cad.onshape.com/documents/ba106cac193c0e2253e1958f/w/3b5db3932a544d9f939ac6a6/e/5c6acb3ff696b03bdf018f3b

## Expected files

- `wildrobot.xml` (MJCF)
- `config.json` (Onshape export config for v2; JSONC)
- `scene_flat_terrain.xml` (training scene; includes `wildrobot.xml`)
- `keyframes.xml` (poses; included by scene)
- `mujoco_robot_config.json` (generated; do not hand-edit)
- `assets/` (meshes; referenced by `wildrobot.xml` compiler `meshdir="assets"`)

## How to (re)generate mujoco_robot_config.json

From repo root:

```bash
cd assets/v2
uv run python ../post_process.py wildrobot.xml
```

This writes `assets/v2/mujoco_robot_config.json` next to the MJCF and (if available) runs a basic validation.

## Digital-twin realism profiles (`v0.19.1`)

Versioned realism parameters for actuator and sensor modeling live here:

- `realism_profile_v0.19.1.json`: typed baseline profile with actuator delay/backlash/frictionloss/armature/effective output limits and sensor noise/latency/dropout parameters
- `realism_profiles.json`: profile registry and default selector

These files are intentionally JSON and diff-friendly so SysID updates can be reviewed in git.

## Actuator order (ABI)

The actuator order in `wildrobot.xml` must be deterministic for policy/action index stability.

- Canonical order lives in `assets/v2/actuator_order.txt` (one name per line).
- After Onshape export, run:
  - `python3 ../reorder_actuators.py --xml wildrobot.xml --order actuator_order.txt`
  - then `python3 ../post_process.py wildrobot.xml`

The repo `assets/update_xml.sh --version v2` does this automatically.
