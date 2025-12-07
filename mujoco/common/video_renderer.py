"""Video rendering utility for WildRobot policies."""

import functools
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import jax
import mediapy as media
import mujoco


class CheckpointInferenceFunction:
    """Inference function callable from checkpoint parameters.

    This class wraps policy network parameters and provides a callable
    interface for deterministic policy inference.
    """

    def __init__(self, policy_network, normalizer_params, policy_params, action_dim):
        """Initialize inference function.

        Args:
            policy_network: Policy network instance
            normalizer_params: Normalizer parameters from checkpoint
            policy_params: Policy parameters from checkpoint
            action_dim: Action dimension size
        """
        self.policy_network = policy_network
        self.normalizer_params = normalizer_params
        self.policy_params = policy_params
        self.action_dim = action_dim

    def __call__(self, obs, rng):
        """Run policy inference - deterministic (use mean).

        Args:
            obs: Observation array
            rng: Random number generator key (unused for deterministic inference)

        Returns:
            Tuple of (action, info_dict)
        """
        logits = self.policy_network.apply(
            self.normalizer_params, self.policy_params, obs
        )
        mean_action = logits[:self.action_dim]
        return mean_action, {}


class VideoRenderer:
    """Renders videos of robot policy execution.

    This class handles:
    - Setting up MuJoCo renderer with specified resolution and camera
    - Running policy rollouts and capturing frames
    - Saving videos to disk with metrics
    - Automatic camera fallback if requested camera doesn't exist
    """

    def __init__(
        self,
        env,
        output_dir: Path,
        width: int = 640,
        height: int = 480,
        fps: int = 50,
        camera: str = "track",
        max_steps: int = 600,
    ):
        """Initialize video renderer.

        Args:
            env: Brax environment to render
            output_dir: Directory to save videos
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            camera: Camera name for rendering
            max_steps: Maximum steps per episode
        """
        self.env = env
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.fps = fps
        self.max_steps = max_steps

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate and set camera
        self.camera = self._validate_camera(camera)

        # JIT compile environment functions
        self.jit_env_reset = jax.jit(env.reset)
        self.jit_env_step = jax.jit(env.step)

    @classmethod
    def from_config(
        cls,
        env,
        cfg: dict,
        output_dir: Path,
        camera: str = "track",
        max_steps: int = 600,
    ):
        """Create VideoRenderer from config dict.

        Args:
            env: Brax environment to render
            cfg: Config dict with rendering settings
            output_dir: Directory to save videos
            camera: Camera name for rendering
            max_steps: Maximum steps per episode

        Returns:
            VideoRenderer instance
        """
        render_width = cfg["rendering"]["render_width"]
        render_height = cfg["rendering"]["render_height"]
        fps = int(1.0 / cfg["env"]["ctrl_dt"])

        return cls(
            env=env,
            output_dir=output_dir,
            width=render_width,
            height=render_height,
            fps=fps,
            camera=camera,
            max_steps=max_steps,
        )

    @staticmethod
    def create_inference_from_checkpoint(params, cfg, env):
        """Create inference function from checkpoint parameters.

        Args:
            params: Checkpoint parameters (normalizer, policy, value)
            cfg: Config dict with network settings
            env: Environment instance (for observation/action sizes)

        Returns:
            Callable inference function
        """
        from brax.training.agents.ppo import networks as ppo_networks

        # Constants for params structure
        PARAMS_NORMALIZER_INDEX = 0
        PARAMS_POLICY_INDEX = 1

        # Build network
        network_config = cfg["network"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=network_config["policy_hidden_layers"],
            value_hidden_layer_sizes=network_config["value_hidden_layers"],
        )

        ppo_networks_tuple = network_factory(env.observation_size, env.action_size)
        policy_network = ppo_networks_tuple.policy_network

        # Extract params
        normalizer_params = params[PARAMS_NORMALIZER_INDEX]
        policy_params = params[PARAMS_POLICY_INDEX]

        # Create and return inference function
        return CheckpointInferenceFunction(
            policy_network=policy_network,
            normalizer_params=normalizer_params,
            policy_params=policy_params,
            action_dim=env.action_size,
        )

    def _validate_camera(self, requested_camera: str) -> Optional[str]:
        """Validate camera exists in model, fallback to first available or None.

        Args:
            requested_camera: Requested camera name

        Returns:
            Valid camera name or None (will use default free camera)
        """
        available_cameras = [
            self.env.mj_model.camera(i).name
            for i in range(self.env.mj_model.ncam)
        ]

        if requested_camera in available_cameras:
            return requested_camera

        if available_cameras:
            print(f"⚠️  Camera '{requested_camera}' not found. Using '{available_cameras[0]}' instead.")
            print(f"    Available cameras: {', '.join(available_cameras)}")
            return available_cameras[0]

        print(f"⚠️  No named cameras found in model. Using default free camera.")
        return None

    def render_video(
        self,
        inference_fn: Callable,
        rng: jax.Array,
        video_name: str = "video",
    ) -> Tuple[Path, dict]:
        """Render a single video of policy execution.

        Args:
            inference_fn: Policy inference function (obs, rng) -> (action, info)
            rng: JAX random key
            video_name: Base name for video file (will add timestamp and .mp4)

        Returns:
            Tuple of (video_path, episode_stats)
            episode_stats contains: {
                'length': int,
                'velocity': float,
                'height': float,
                'distance': float,
                'success': bool,
            }
        """
        start_time = time.time()

        # Reset environment
        rng, reset_rng = jax.random.split(rng)
        state = self.jit_env_reset(reset_rng)

        # Setup renderer
        renderer = mujoco.Renderer(
            self.env.mj_model,
            height=self.height,
            width=self.width
        )
        print(f"  ✓ Renderer initialized ({self.width}x{self.height})")

        # Create MuJoCo data for rendering
        mj_data = mujoco.MjData(self.env.mj_model)

        # Rollout episode and capture frames
        frames = []
        for step_idx in range(self.max_steps):
            # Update MuJoCo data
            mj_data.qpos[:] = state.data.qpos
            mj_data.qvel[:] = state.data.qvel
            mujoco.mj_forward(self.env.mj_model, mj_data)

            # Render frame
            renderer.update_scene(mj_data, camera=self.camera)
            frame = renderer.render()
            frames.append(frame)

            # Step environment
            rng, step_rng = jax.random.split(rng)
            act_rng, _ = jax.random.split(step_rng)
            obs = state.obs
            action, _ = inference_fn(obs, act_rng)
            state = self.jit_env_step(state, action)

            # Check if done
            if state.done:
                print(f"  ✓ Episode ended at step {step_idx + 1}/{self.max_steps}")
                break

        # Close renderer
        renderer.close()

        # Extract episode statistics
        episode_length = len(frames)
        final_velocity = float(state.metrics.get("forward_velocity", 0.0))
        final_height = float(state.metrics.get("height", 0.0))
        distance = float(state.metrics.get("distance_walked", 0.0))
        success = bool(state.metrics.get("success", False))

        episode_stats = {
            'length': episode_length,
            'velocity': final_velocity,
            'height': final_height,
            'distance': distance,
            'success': success,
        }

        print(f"  ✓ Captured {episode_length} frames")
        print(f"    Velocity: {final_velocity:.3f} m/s")
        print(f"    Height: {final_height:.3f} m")
        print(f"    Distance: {distance:.3f} m")

        # Save video
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_path = self.output_dir / f"{video_name}_{timestamp}.mp4"

        if frames:
            print(f"  ✓ Saving to {video_path.name}...")
            media.write_video(str(video_path), frames, fps=self.fps)

            # Verify
            if video_path.exists():
                file_size_mb = video_path.stat().st_size / (1024 * 1024)
                elapsed = time.time() - start_time
                print(f"  ✓ Saved: {video_path.name} ({file_size_mb:.2f} MB) in {elapsed:.1f}s")
            else:
                print(f"  ✗ ERROR: Video file not created!")
                raise RuntimeError(f"Failed to create video file: {video_path}")
        else:
            print(f"  ✗ WARNING: No frames captured!")
            raise RuntimeError("No frames captured during episode")

        return video_path, episode_stats

    def render_multiple_videos(
        self,
        inference_fn: Callable,
        rng: jax.Array,
        num_videos: int = 3,
        video_prefix: str = "eval_video",
    ) -> list:
        """Render multiple videos with different random seeds.

        Args:
            inference_fn: Policy inference function (obs, rng) -> (action, info)
            rng: JAX random key
            num_videos: Number of videos to render
            video_prefix: Prefix for video filenames

        Returns:
            List of (video_path, episode_stats) tuples
        """
        results = []

        print("="*80)
        print("Starting video rendering...")
        print("="*80 + "\n")

        for video_idx in range(num_videos):
            print(f"Rendering video {video_idx + 1}/{num_videos}...")

            try:
                # Split RNG for this video
                rng, video_rng = jax.random.split(rng)

                # Render video
                video_path, episode_stats = self.render_video(
                    inference_fn=inference_fn,
                    rng=video_rng,
                    video_name=f"{video_prefix}_{video_idx}",
                )

                results.append((video_path, episode_stats))
                print()  # Blank line between videos

            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                print()
                continue

        print("="*80)
        print(f"Rendering complete! Videos saved to: {self.output_dir}")
        print("="*80)

        return results
