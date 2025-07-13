import fcntl
import os
import time
import zipfile

from huggingface_hub import hf_hub_download


def extract_videos_from_single_process(dataset_name: str, video_cache_dir: str) -> str:
    """Extract videos from dataset zip if not already extracted from main process."""
    videos_dir = os.path.join(video_cache_dir, "videos")
    lock_file = os.path.join(video_cache_dir, ".download_lock")
    expected_zip_path = os.path.join(video_cache_dir, "videos.zip")

    # Check if videos directory already exists and has content
    if os.path.exists(videos_dir) and os.listdir(videos_dir):
        print(f"Videos already extracted at: {videos_dir}")
        return videos_dir

    # Use file-based locking for distributed coordination
    os.makedirs(video_cache_dir, exist_ok=True)

    # Try to acquire lock (only one process will succeed)
    try:
        with open(lock_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Double-check after acquiring lock
            if os.path.exists(videos_dir) and os.listdir(videos_dir):
                print(f"Videos already extracted at: {videos_dir}")
                return videos_dir

            # Download and extract
            print("Downloading videos.zip from HuggingFace dataset repository...")
            zip_path = hf_hub_download(
                repo_id=dataset_name,
                filename="videos.zip",
                repo_type="dataset",
                local_dir=video_cache_dir,
                local_dir_use_symlinks=False,
            )

            print(f"Extracting videos from {zip_path}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(video_cache_dir)

            os.remove(zip_path)
            print(f"Videos extracted successfully to: {videos_dir}")

            # Count and print extracted videos
            train_dir = os.path.join(videos_dir, "train")
            val_dir = os.path.join(videos_dir, "val")

            train_count = (
                len([f for f in os.listdir(train_dir) if f.endswith(".mp4")]) if os.path.exists(train_dir) else 0
            )
            val_count = len([f for f in os.listdir(val_dir) if f.endswith(".mp4")]) if os.path.exists(val_dir) else 0

            print(f"Extraction complete! Total videos - Train: {train_count}, Validation: {val_count}")

    except (IOError, OSError):
        # Lock acquisition failed, wait for other process to finish
        print("Another process is downloading videos, waiting...")
        while not (os.path.exists(videos_dir) and os.listdir(videos_dir) and not os.path.exists(expected_zip_path)):
            videos_exist = os.path.exists(videos_dir)
            has_content = videos_exist and os.listdir(videos_dir)
            zip_deleted = not os.path.exists(expected_zip_path)
            print(
                f"Waiting for extraction to complete... "
                f"(checking videos_dir: {videos_exist}, has_content: {has_content}, "
                f"zip_deleted: {zip_deleted})"
            )
            time.sleep(5)
        print("Videos ready!")

    return videos_dir


def get_system_message(use_system_message: bool) -> str:
    if use_system_message:
        system_message = (
            "You are an expert and intelligent question answering agent. "
            "You will be shown a video that was collected by a robot yesterday while navigating around a house "
            "and picking and placing objects. Each frame in the video has a unique frame index in the top left corner "
            "of the video along with the time of day information. Your job is to help the robot complete a task today "
            "by looking at the video and finding the frame indices that the robot should move to. "
            "Note: The robot uses a magic grasp action to pick up an object, where a gripper goes close to the object "
            "and the object gets magically picked up. When deciding which frame indices to choose, "
            "make sure you choose the frame indices that are closest to the object/place."
        )
    else:
        system_message = ""

    return system_message


def extract_assistant_response(text):
    """Extract only the assistant's response from the full model output."""
    if "assistant\n" in text:
        return text.split("assistant\n", 1)[1].strip()
    return text  # Return original text if pattern not found
