# assets/v2 (WildRobot full body with arms)

This folder is the **v2** robot asset variant (full body with arms).

- Onshape document (assembly): `WildRobot`
  - Link: https://cad.onshape.com/documents/ba106cac193c0e2253e1958f/w/3b5db3932a544d9f939ac6a6/e/5c6acb3ff696b03bdf018f3b

## Expected files

- `wildrobot.xml` (MJCF)
- `config.json` (Onshape export config for v2; JSONC)
- `scene_flat_terrain.xml` (training scene; includes `wildrobot.xml`)
- `keyframes.xml` (poses; included by scene)
- `robot_config.yaml` (generated; do not hand-edit)
- `assets/` (meshes; referenced by `wildrobot.xml` compiler `meshdir="assets"`)

## How to (re)generate robot_config.yaml

From repo root:

```bash
cd assets/v2
uv run python ../post_process.py wildrobot.xml
```

This writes `assets/v2/robot_config.yaml` next to the MJCF and (if available) runs a basic validation.
