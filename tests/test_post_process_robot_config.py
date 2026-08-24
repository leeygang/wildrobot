import json
from pathlib import Path

from assets.post_process import generate_robot_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_generate_robot_config_refreshes_derived_world_axes(tmp_path: Path) -> None:
    output_path = tmp_path / "mujoco_robot_config.json"
    output_path.write_text(
        json.dumps(
            {
                "actuated_joint_specs": [
                    {
                        "name": "left_elbow_pitch",
                        "max_velocity": 7.5,
                        "init_world_axis": [1.0, 0.0, 0.0],
                    }
                ]
            }
        )
    )

    config = generate_robot_config(
        str(PROJECT_ROOT / "assets/v2/wildrobot.xml"),
        str(output_path),
    )
    elbow = next(
        spec
        for spec in config["actuated_joint_specs"]
        if spec["name"] == "left_elbow_pitch"
    )

    assert elbow["max_velocity"] == 7.5
    assert elbow["init_world_axis"] != [1.0, 0.0, 0.0]
    assert elbow["init_world_axis"][1] < -0.9
