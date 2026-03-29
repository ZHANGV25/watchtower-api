import subprocess
import yaml
import os
import glob
import shutil
import json
import cv2
import pandas as pd

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def get_frame_files(image_directory):
    """Return sorted image frames from a directory."""
    if not os.path.isdir(image_directory):
        raise FileNotFoundError(f"Image directory not found: {image_directory}")

    frame_files = sorted(
        file_name
        for file_name in glob.glob(os.path.join(image_directory, "*"))
        if os.path.isfile(file_name) and file_name.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not frame_files:
        raise ValueError(
            f"No image frames found in {image_directory}. "
            "Expected .png, .jpg, .jpeg, .bmp, or .webp files."
        )

    return frame_files


def resolve_input_source(input_path, output_dir, extracted_frames_dirname):
    """Accept either a video file or a frame directory and return frame metadata."""
    absolute_input_path = os.path.abspath(input_path)

    if os.path.isdir(absolute_input_path):
        frame_files = get_frame_files(absolute_input_path)
        return {
            "input_type": "frames",
            "source_path": absolute_input_path,
            "frame_directory": absolute_input_path,
            "frame_files": frame_files,
            "fps": 30.0,
        }

    if not os.path.isfile(absolute_input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not absolute_input_path.lower().endswith(VIDEO_EXTENSIONS):
        raise ValueError(
            f"Unsupported input file: {input_path}. "
            "Expected a video file or a directory of frames."
        )

    extracted_frames_dir = os.path.abspath(
        os.path.join(output_dir, extracted_frames_dirname)
    )
    if os.path.exists(extracted_frames_dir):
        shutil.rmtree(extracted_frames_dir)
    os.makedirs(extracted_frames_dir, exist_ok=True)

    capture = cv2.VideoCapture(absolute_input_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_path = os.path.join(extracted_frames_dir, f"{frame_index + 1:05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_index += 1

    capture.release()

    if frame_index == 0:
        raise ValueError(f"No frames could be extracted from video: {input_path}")

    frame_files = get_frame_files(extracted_frames_dir)
    return {
        "input_type": "video",
        "source_path": absolute_input_path,
        "frame_directory": extracted_frames_dir,
        "frame_files": frame_files,
        "fps": float(fps),
    }


def prepare_pure_style_dataset(frame_files, output_dir):
    """Stage arbitrary image frames into a PURE-like folder layout."""
    dataset_root = os.path.join(output_dir, "pure_like_dataset")
    session_name = "01-01"
    session_dir = os.path.join(dataset_root, session_name)
    staged_frames_dir = os.path.join(session_dir, session_name)

    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)

    os.makedirs(staged_frames_dir, exist_ok=True)

    for index, source_path in enumerate(frame_files, start=1):
        target_path = os.path.join(staged_frames_dir, f"{index:05d}.png")
        shutil.copy2(source_path, target_path)

    return dataset_root

def run_rppg_toolbox(input_path, output_dir="rppg_results"):
    """
    Configures and runs the rPPG-Toolbox for patient assessment.
    Accepts either a video file or a directory of frames.
    """
    print(f"--> Configuring rPPG-Toolbox for Unsupervised Assessment...")
    
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "patient_infer_config.yaml")
    input_info = resolve_input_source(input_path, output_dir, "rppg_toolbox_frames")

    # 1. Generate a toolbox-compatible YAML configuration.
    methods = ["POS", "CHROM"]
    config_dict = {
        "BASE": [""],
        "TOOLBOX_MODE": "unsupervised_method",
        "INFERENCE": {
            "EVALUATION_METHOD": "FFT",
            "EVALUATION_WINDOW": {
                "USE_SMALLER_WINDOW": False,
                "WINDOW_SIZE": 10,
            },
        },
        "LOG": {
            "PATH": output_dir
        },
        "UNSUPERVISED": {
            "METHOD": methods,
            "METRICS": [],
            "EXTRACT_ONLY": True,
            "OUTPUT_SAVE_DIR": output_dir,
            "DATA": {
                "FS": round(input_info["fps"]),
                "DATA_PATH": input_info["frame_directory"],
                "CACHED_PATH": os.path.join(output_dir, "cached_data"),
                "EXP_DATA_NAME": "patient_assessment",
                "DATASET": "Upload",
                "DO_PREPROCESS": True,
                "DATA_FORMAT": "NDHWC",
                "BEGIN": 0.0,
                "END": 1.0,
                "PREPROCESS": {
                    "USE_PSUEDO_PPG_LABEL": False,
                    "DATA_TYPE": ["Raw"],
                    "DATA_AUG": ["None"],
                    "LABEL_TYPE": "Raw",
                    "DO_CHUNK": True,
                    "CHUNK_LENGTH": 300,
                    "CROP_FACE": {
                        "DO_CROP_FACE": True,
                        "BACKEND": "HC",
                        "USE_LARGE_FACE_BOX": True,
                        "LARGE_BOX_COEF": 1.5,
                        "DETECTION": {
                            "DO_DYNAMIC_DETECTION": False,
                            "DYNAMIC_DETECTION_FREQUENCY": round(input_info["fps"]),
                            "USE_MEDIAN_FACE_BOX": False,
                        },
                    },
                    "RESIZE": {
                        "H": 72,
                        "W": 72,
                    },
                },
            },
        }
    }

    # Save the YAML file
    with open(config_path, 'w') as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)

    print(f"--> YAML config saved to {config_path}. Executing rPPG-Toolbox...")

    # 2. Execute the toolbox via its dedicated interpreter.
    toolbox_python = os.environ.get(
        "RPPG_TOOLBOX_PYTHON",
        "/opt/anaconda3/envs/rppg-toolbox/bin/python",
    )
    toolbox_cwd = os.path.join(os.getcwd(), "rPPG-Toolbox")
    command = [toolbox_python, "main.py", "--config_file", config_path]
    
    try:
        # Run the process and wait for it to finish
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=toolbox_cwd,
        )
        print("--> rPPG-Toolbox execution completed successfully.")

        method_results = {}
        for method in methods:
            matching_files = glob.glob(
                os.path.join(output_dir, "**", f"rppg_toolbox_{method.lower()}_results.json"),
                recursive=True,
            )
            if matching_files:
                with open(matching_files[-1], "r", encoding="utf-8") as result_file:
                    method_results[method] = json.load(result_file)

        if not method_results:
            return {"error": "Toolbox ran, but no JSON extraction results were found."}

        return {
            "toolbox": "rPPG-Toolbox",
            "fps": input_info["fps"],
            "input_type": input_info["input_type"],
            "frame_count": len(input_info["frame_files"]),
            "methods": method_results,
        }

    except subprocess.CalledProcessError as e:
        print(f"Error running rPPG-Toolbox. \nStandard Error: {e.stderr}")
        return None


def run_rppg2_toolbox(input_path, output_dir="rppg_results"):
    """
    Runs mcd_rppg unsupervised POS/CHROM inference on a video or folder of frames.
    """
    print("--> Configuring mcd_rppg for Unsupervised Assessment...")

    os.makedirs(output_dir, exist_ok=True)
    input_info = resolve_input_source(input_path, output_dir, "mcd_rppg_frames")
    frame_files = input_info["frame_files"]
    result_path = os.path.join(output_dir, "mcd_rppg_results.json")
    runner_path = os.path.join(output_dir, "mcd_rppg_runner.py")

    runner_code = """
import json
import os
import sys

import cv2
import numpy as np

repo_root = sys.argv[1]
image_dir = sys.argv[2]
result_path = sys.argv[3]
fps = float(sys.argv[4])

sys.path.insert(0, repo_root)

from mcd_rppg.rppglib.unsupervised import POS_WANG, chrome
from scipy import signal

def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def calculate_fft_hr(ppg_signal, fs=30, low_pass=0.5, high_pass=3.5):
    ppg_signal = np.expand_dims(ppg_signal, 0)
    nfft = _next_power_of_2(ppg_signal.shape[1])
    frequencies, power = signal.periodogram(ppg_signal, fs=fs, nfft=nfft, detrend=False)
    mask = (frequencies >= low_pass) & (frequencies <= high_pass)
    masked_frequencies = frequencies[mask]
    masked_power = power[:, mask]
    return masked_frequencies[np.argmax(masked_power, axis=1)][0] * 60

supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
frame_files = sorted(
    os.path.join(image_dir, file_name)
    for file_name in os.listdir(image_dir)
    if file_name.lower().endswith(supported_extensions)
)

frames = []
for frame_path in frame_files:
    frame = cv2.imread(frame_path)
    if frame is None:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

if not frames:
    raise ValueError(f"No readable frames found in {image_dir}")

video = np.asarray(frames, dtype=np.uint8)
rgb = video.reshape(video.shape[0], -1, 3).mean(axis=1).astype("float32")

pos_bvp = POS_WANG(rgb, fps)
chrom_bvp = chrome(rgb, fps)

results = {
    "toolbox": "mcd_rppg",
    "fps": fps,
    "frame_count": int(video.shape[0]),
    "pos_hr_bpm": float(calculate_fft_hr(pos_bvp, fs=fps)),
    "chrom_hr_bpm": float(calculate_fft_hr(chrom_bvp, fs=fps)),
}

with open(result_path, "w", encoding="utf-8") as result_file:
    json.dump(results, result_file, indent=2)

print(json.dumps(results))
""".strip()

    with open(runner_path, "w", encoding="utf-8") as runner_file:
        runner_file.write(runner_code)

    print(
        f"--> Found {len(frame_files)} frames. Executing mcd_rppg "
        f"and saving results to {result_path}..."
    )

    mcd_rppg_python = os.environ.get(
        "MCD_RPPG_PYTHON",
        os.environ.get(
            "RPPG_TOOLBOX_PYTHON",
            "/opt/anaconda3/envs/rppg-toolbox/bin/python",
        ),
    )

    command = [
        mcd_rppg_python,
        runner_path,
        os.getcwd(),
        input_info["frame_directory"],
        result_path,
        str(input_info["fps"]),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        print("--> mcd_rppg execution completed successfully.")
        return json.loads(result.stdout.strip().splitlines()[-1])
    except subprocess.CalledProcessError as e:
        print(f"Error running mcd_rppg. \nStandard Error: {e.stderr}")
        return None


def run_gaitanalyzer(input_path, output_dir="rppg_results"):
    """
    Runs gaitanalyzer on a video file or a folder of frames.
    """
    print("--> Configuring gaitanalyzer for gait assessment...")

    os.makedirs(output_dir, exist_ok=True)
    input_info = resolve_input_source(input_path, output_dir, "gaitanalyzer_frames")
    frame_files = input_info["frame_files"]
    result_path = os.path.join(output_dir, "gaitanalyzer_results.json")
    runner_path = os.path.join(output_dir, "gaitanalyzer_runner.py")

    runner_code = """
import json
import os
import sys

import cv2

repo_root = sys.argv[1]
image_dir = sys.argv[2]
result_path = sys.argv[3]
fps = float(sys.argv[4])

repo_path = os.path.join(repo_root, "gaitanalyzer")
sys.path.insert(0, repo_path)
os.chdir(repo_path)

from gait_analysis import GaitAnalysis

supported_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
frame_files = sorted(
    os.path.join(image_dir, file_name)
    for file_name in os.listdir(image_dir)
    if file_name.lower().endswith(supported_extensions)
)

frames = []
frame_size = None
for frame_path in frame_files:
    frame = cv2.imread(frame_path)
    if frame is None:
        continue
    if frame_size is None:
        frame_size = (frame.shape[1], frame.shape[0])
    elif (frame.shape[1], frame.shape[0]) != frame_size:
        frame = cv2.resize(frame, frame_size)
    frames.append(frame)

if not frames:
    raise ValueError(f"No readable frames found in {image_dir}")

input_dir = os.path.join(repo_root, "rppg_results", "gaitanalyzer_input")
os.makedirs(input_dir, exist_ok=True)
input_video_path = os.path.join(input_dir, "patient_walk.mp4")

writer = cv2.VideoWriter(
    input_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    frame_size,
)
for frame in frames:
    writer.write(frame)
writer.release()

analysis = GaitAnalysis(
    input_video_path,
    model_path=os.path.join(repo_path, "model", "pose_landmarker_heavy.task"),
)
output_video_path, df, result_text, _ = analysis.process_video()

csv_path = os.path.join(repo_root, "rppg_results", "gaitanalyzer_metrics.csv")
df.to_csv(csv_path, index=False)

summary = {}
for key, value in df.mean(numeric_only=True).dropna().to_dict().items():
    summary[key] = float(value)

results = {
    "toolbox": "gaitanalyzer",
    "frame_count": len(frames),
    "fps": fps,
    "input_video_path": input_video_path,
    "annotated_video_path": output_video_path,
    "metrics_csv_path": csv_path,
    "summary": summary,
    "result_text": result_text,
}

with open(result_path, "w", encoding="utf-8") as result_file:
    json.dump(results, result_file, indent=2)

print(json.dumps(results))
""".strip()

    with open(runner_path, "w", encoding="utf-8") as runner_file:
        runner_file.write(runner_code)

    print(
        f"--> Found {len(frame_files)} frames. Executing gaitanalyzer "
        f"and saving results to {result_path}..."
    )

    gaitanalyzer_python = os.environ.get(
        "GAIT_ANALYZER_PYTHON",
        "/opt/anaconda3/envs/gaitanalyzer/bin/python",
    )

    command = [
        gaitanalyzer_python,
        runner_path,
        os.getcwd(),
        input_info["frame_directory"],
        result_path,
        str(input_info["fps"]),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        print("--> gaitanalyzer execution completed successfully.")
        return json.loads(result.stdout.strip().splitlines()[-1])
    except subprocess.CalledProcessError as e:
        print(f"Error running gaitanalyzer. \nStandard Error: {e.stderr}")
        return None


def run_gaitanalyzer2(input_path, output_dir="rppg_results"):
    """
    Summarizes the static GaitAnalysisVLM repository contents.

    This repository does not ship an executable inference pipeline in this clone,
    so this wrapper records repository metadata instead of attempting analysis.
    """
    print("--> Inspecting GaitAnalysisVLM repository...")

    os.makedirs(output_dir, exist_ok=True)
    repo_path = os.path.join(os.getcwd(), "GaitAnalysisVLM")
    result_path = os.path.join(output_dir, "gaitanalyzer2_results.json")

    repo_exists = os.path.isdir(repo_path)
    html_files = []
    static_files = []

    if repo_exists:
        html_files = sorted(glob.glob(os.path.join(repo_path, "*.html")))
        static_files = sorted(glob.glob(os.path.join(repo_path, "static", "**", "*"), recursive=True))
        static_files = [path for path in static_files if os.path.isfile(path)]

    input_exists = os.path.exists(input_path)
    input_type = "missing"
    frame_count = 0
    fps = None
    if input_exists:
        try:
            input_info = resolve_input_source(input_path, output_dir, "gaitanalyzer2_frames")
            input_type = input_info["input_type"]
            frame_count = len(input_info["frame_files"])
            fps = input_info["fps"]
        except (ValueError, FileNotFoundError):
            frame_count = 0

    results = {
        "toolbox": "GaitAnalysisVLM",
        "repo_path": repo_path,
        "repo_present": repo_exists,
        "input_path": input_path,
        "input_exists": input_exists,
        "input_type": input_type,
        "frame_count": frame_count,
        "fps": fps,
        "html_file_count": len(html_files),
        "static_asset_count": len(static_files),
        "runnable_inference_found": False,
        "status": (
            "Repository cloned, but this checkout only contains static HTML assets "
            "and documentation. No runnable inference entrypoint was found."
        ),
        "recommended_next_step": (
            "Use the GaVA-CLIP implementation referenced by the repository README "
            "if you want an executable gait-analysis model."
        ),
    }

    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)

    print("--> GaitAnalysisVLM inspection completed.")
    return results

# --- Example Execution ---
if __name__ == "__main__":
    # Point this to either a video file or a folder containing extracted frames.
    patient_input = "./data/patient_001_walk.mp4"
    
    vitals = run_rppg_toolbox(patient_input)
    print("\nExtracted Vitals Data:")
    print(vitals)

    vitals2 = run_rppg2_toolbox(patient_input)
    print("\nExtracted Vitals Data (mcd_rppg):")
    print(vitals2)

    gait_vitals = run_gaitanalyzer(patient_input)
    print("\nExtracted Gait Data:")
    print(gait_vitals)

    gait_vitals2 = run_gaitanalyzer2(patient_input)
    print("\nExtracted Gait Data (GaitAnalysisVLM):")
    print(gait_vitals2)