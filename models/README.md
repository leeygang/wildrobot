# WildRobot Models

This directory contains MuJoCo MJCF XML models for the WildRobot humanoid.

## Structure

- `wildrobot_v1/` - Current robot model
- `wildrobot_v2/` - Future iterations
- `assets/` - Shared meshes, textures, collision shapes

## Usage

Test model in MuJoCo viewer:
```bash
python -m mujoco.viewer models/wildrobot_v1/wildrobot.xml
```
