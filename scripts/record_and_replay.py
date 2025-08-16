"""
Record and replay episodes with the BiSO1016AxisFollower robot and its teleoperator.
how to use:
1. Run the script to record episodes:
    python scripts/record_and_replay.py --mode record
2. Replay a specific episode:
    python scripts/record_and_replay.py --mode replay
3. Record and replay in one go:
    python scripts/record_and_replay.py --mode both
    or
    python scripts/record_and_replay.py
"""

import time
import argparse

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

from lerobot.robots.bi_so101_6axis_follower import BiSO1016AxisFollower, BiSO1016AxisFollowerConfig
from lerobot.teleoperators.bi_so101_6axis_leader.config_bi_so101_6axis_leader import BiSO1016AxisLeaderConfig
from lerobot.teleoperators.bi_so101_6axis_leader.bi_so101_6axis_leader import BiSO1016AxisLeader

from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

from lerobot.record import record_loop

ROBOT_LEFT_ARM_PORT = "/dev/ttyACM2"
ROBOT_RIGHT_ARM_PORT = "/dev/ttyACM0"
TELEOP_LEFT_ARM_PORT = "/dev/ttyACM3"
TELEOP_RIGHT_ARM_PORT = "/dev/ttyACM1"

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 180
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "open campus task"

DATASET_REPO_ID = "SSatoya/record_open_campus_bi_so101_6axis"

REPLAY_EPISODE_IDX = 0  # Episode index for replay

def record_episode():
    """
    Record an episode with the BiSO1016AxisFollower robot and its teleoperator.
    """
    # Create the robot and teleoperator configurations
    camera_config = {
        "left": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=FPS),
        "right": OpenCVCameraConfig(index_or_path="/dev/video2", width=640, height=480, fps=FPS),
    }
    robot_config = BiSO1016AxisFollowerConfig(
        left_arm_port=ROBOT_LEFT_ARM_PORT, 
        right_arm_port=ROBOT_RIGHT_ARM_PORT,
        id="bi_6axis_so101_follower", 
        cameras=camera_config
    )
    teleop_config = BiSO1016AxisLeaderConfig(
        left_arm_port=TELEOP_LEFT_ARM_PORT, 
        right_arm_port=TELEOP_RIGHT_ARM_PORT,
        id="bi_6axis_so101_leader"
    )

    # Initialize the robot and teleoperator
    robot = BiSO1016AxisFollower(robot_config)
    teleop = BiSO1016AxisLeader(teleop_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Create the dataset
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # Initialize the keyboard listener and rerun visualization
    _, events = init_keyboard_listener()
    _init_rerun(session_name="recording")

    # Connect the robot and teleoperator
    robot.connect()
    teleop.connect()

    episode_idx = 0
    while episode_idx < NUM_EPISODES and not events["stop_recording"]:
        log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        # Reset the environment if not stopping or re-recording
        if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode_idx += 1

    # Clean up
    log_say("Stop recording")
    robot.disconnect()
    teleop.disconnect()
    # dataset.push_to_hub()

def replay_episode():
    """
    Replay the recorded episodes from the dataset.
    """
    robot_config = BiSO1016AxisFollowerConfig(
        left_arm_port=ROBOT_LEFT_ARM_PORT,
        right_arm_port=ROBOT_RIGHT_ARM_PORT,
        id="bi_6axis_so101_follower"
    )

    robot = BiSO1016AxisFollower(robot_config)
    robot.connect()

    dataset = LeRobotDataset(
        DATASET_REPO_ID,
        episodes=[REPLAY_EPISODE_IDX]
    )
    actions = dataset.hf_dataset.select_columns("action")

    log_say(f"Replaying episode {REPLAY_EPISODE_IDX}")
    for idx in range(dataset.num_frames):
        t0 = time.perf_counter()

        action= {
            name: float(actions[idx]["action"][i])
            for i, name in enumerate(dataset.features["action"]["names"])
        }
        robot.send_action(action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))
    
    robot.disconnect()
    log_say("Replay finished")

# def main():
#     record_episode()

#     log_say(f"Recording completed. ")
#     time.sleep(2)

#     replay_episode()
def main():
    global REPLAY_EPISODE_IDX 
    parser = argparse.ArgumentParser(description="Record and/or replay robot episodes")
    parser.add_argument(
        "--mode", 
        choices=["record", "replay", "both"], 
        default="both",
        help="Choose mode: record only, replay only, or both (default: both)"
    )
    parser.add_argument(
        "--episode-idx", 
        type=int, 
        default=REPLAY_EPISODE_IDX,
        help=f"Episode index for replay (default: {REPLAY_EPISODE_IDX})"
    )
    
    args = parser.parse_args()

    # global REPLAY_EPISODE_IDX
    
    if args.mode in ["record", "both"]:
        log_say("Starting recording...")
        record_episode()
        
        if args.mode == "both":
            time.sleep(2)
    
    if args.mode in ["replay", "both"]:
        log_say("Starting replay...")
        # Update the global replay episode index
        REPLAY_EPISODE_IDX = args.episode_idx
        replay_episode()
        log_say("Replay completed.")

    print("plz delete this file after use. this command will delete the cache.")
    print("rm -r ~/.cache/huggingface/lerobot/SSatoya/record_open_campus_bi_so101_6axis")

if __name__ == "__main__":
    main()