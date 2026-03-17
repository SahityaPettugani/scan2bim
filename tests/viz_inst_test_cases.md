# viz_inst.py Pipeline Test Cases

## Scope
These test cases cover the CLI and end-to-end pipeline in `viz_inst.py`:
- point-cloud load
- checkpoint loading and BIMNet inference
- optional KNN smoothing
- class separation + instance extraction (RANSAC/DBSCAN)
- output artifact generation (`.ply` + JSON)

## Conventions
- `TC-VIZ-###` = test case identifier
- `UC-VIZ-###` = related use case identifier
- All commands are run from repository root.

---

## TC-VIZ-001
- Identifier: `TC-VIZ-001` (links to `UC-VIZ-001`: run successful pipeline on valid `.ply` and checkpoint)
- Description: Validate that a valid input point cloud and valid checkpoint complete the full pipeline and produce a run folder with expected artifacts.
- Precondition:
  - Python environment has required dependencies (`open3d`, `torch`, `numpy`, `sklearn`, `pyransac3d`, etc.).
  - Input file exists and is readable: `C:\Users\iamsa\Downloads\synth_room_100.ply`.
  - Checkpoint exists and is readable: `C:\Users\iamsa\Downloads\val_best_miou (1).pth`.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --no-vis-instances
    ```
- Expected output:
  - Console includes:
    - `Point Cloud Instantiation Workflow (BIMNet + DBSCAN)`
    - `Run output directory: ...`
    - `Step 3: Extracting BIM Parameters and Saving...`
  - New timestamped output directory under `output_instances\`.
  - Files exist in run directory:
    - `instantiation_summary.json`
    - `all_instances_combined.ply`
    - `bim_reconstruction_data.json`
- Expected postcondition:
  - Process exits with code `0`.
  - `bim_reconstruction_data.json` is valid JSON array.
  - At least one class subfolder exists (if any instance survives cleaning thresholds).
- Execution history (log):
  - Date/time:
  - Tester:
  - Commit/branch:
  - Actual command:
  - Actual output directory:
  - Pass/Fail:
  - Notes:

## TC-VIZ-002
- Identifier: `TC-VIZ-002` (links to `UC-VIZ-002`: reject unsupported input extension)
- Description: Verify unsupported input format is rejected.
- Precondition:
  - File exists with unsupported extension, e.g. `C:\temp\dummy.xyz`.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\temp\dummy.xyz" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --no-vis-instances
    ```
- Expected output:
  - Error containing `Unsupported file format`.
- Expected postcondition:
  - Process exits non-zero.
  - No new run folder is created.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-003
- Identifier: `TC-VIZ-003` (links to `UC-VIZ-003`: fail on missing input file)
- Description: Verify missing input file fails safely.
- Precondition:
  - Input path does not exist.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\does_not_exist\missing.ply" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --no-vis-instances
    ```
- Expected output:
  - Open3D/file-load failure message.
- Expected postcondition:
  - Process exits non-zero.
  - No `bim_reconstruction_data.json` created for this run.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-004
- Identifier: `TC-VIZ-004` (links to `UC-VIZ-004`: fail on missing checkpoint)
- Description: Verify missing checkpoint fails at model loading.
- Precondition:
  - Input point cloud exists.
  - Checkpoint path does not exist.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\does_not_exist\bad_checkpoint.pth" --no-vis-instances
    ```
- Expected output:
  - Error from `torch.load` (file not found/read error).
- Expected postcondition:
  - Process exits non-zero.
  - Run directory may be created before model load, but output artifacts are incomplete or missing.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-005
- Identifier: `TC-VIZ-005` (links to `UC-VIZ-005`: CPU execution path)
- Description: Validate `--cpu` forces CPU path and still completes successfully.
- Precondition:
  - Same as successful run prerequisites.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --cpu --no-vis-instances
    ```
- Expected output:
  - Run completes; no CUDA-required failure.
- Expected postcondition:
  - Output files exist as in TC-VIZ-001.
  - Process exits code `0`.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-006
- Identifier: `TC-VIZ-006` (links to `UC-VIZ-006`: smoothing disabled behavior)
- Description: Verify `--no-smooth` disables KNN smoothing step.
- Precondition:
  - Valid input and checkpoint.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --no-smooth --no-vis-instances
    ```
- Expected output:
  - Console contains `Step 0.5: KNN smoothing disabled (--no-smooth).`
- Expected postcondition:
  - Pipeline completes (assuming valid model/input).
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-007
- Identifier: `TC-VIZ-007` (links to `UC-VIZ-007`: auto-skip smoothing for large cloud)
- Description: Verify smoothing is skipped when point count exceeds `--smooth-max-points`.
- Precondition:
  - Input with point count higher than chosen threshold.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --smooth-max-points 10 --no-vis-instances
    ```
- Expected output:
  - Console includes `Skipping KNN smoothing because point count (...) exceeds --smooth-max-points (10).`
- Expected postcondition:
  - Pipeline continues after skip and produces outputs.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-008
- Identifier: `TC-VIZ-008` (links to `UC-VIZ-008`: multiple checkpoint ensemble)
- Description: Validate support for multiple `--checkpoint` arguments and successful averaged inference.
- Precondition:
  - Two compatible checkpoint files exist.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\path\ckpt1.pth" --checkpoint "C:\path\ckpt2.pth" --no-vis-instances
    ```
- Expected output:
  - Console shows two `Loading checkpoint:` lines.
- Expected postcondition:
  - Pipeline completes and output artifacts are generated.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-009
- Identifier: `TC-VIZ-009` (links to `UC-VIZ-009`: output directory naming and run isolation)
- Description: Verify each run creates a unique timestamped subdirectory under `--output_dir`.
- Precondition:
  - Valid input/checkpoint.
- Inputs:
  - Command (run twice):
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --output_dir "output_instances" --no-vis-instances
    ```
- Expected output:
  - Two different `Run output directory:` values.
- Expected postcondition:
  - No overwrite between runs.
  - Each run has its own `bim_reconstruction_data.json`.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-010
- Identifier: `TC-VIZ-010` (links to `UC-VIZ-010`: output schema sanity for BIM JSON)
- Description: Validate `bim_reconstruction_data.json` object schema for produced elements.
- Precondition:
  - Successful run exists (from TC-VIZ-001).
- Inputs:
  - Artifact under test: `<run_dir>\bim_reconstruction_data.json`.
- Expected output:
  - File parses as JSON list.
  - For each object:
    - Has keys: `id`, `type`, `height`, `thickness`, `geometry`.
    - `geometry` has keys: `start_x`, `start_y`, `start_z`, `end_x`, `end_y`, `end_z`.
    - Numeric fields are numbers.
- Expected postcondition:
  - Schema checks pass for all entries.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-011
- Identifier: `TC-VIZ-011` (links to `UC-VIZ-011`: deterministic artifact presence with visualization disabled)
- Description: Confirm headless-friendly mode (`--no-vis-instances`) does not block output generation.
- Precondition:
  - Environment may not support GUI display.
- Inputs:
  - Command:
    ```powershell
    python viz_inst.py --input_file "C:\Users\iamsa\Downloads\synth_room_100.ply" --checkpoint "C:\Users\iamsa\Downloads\val_best_miou (1).pth" --no-vis-instances
    ```
- Expected output:
  - No Open3D visualization window is opened.
  - Save steps complete normally.
- Expected postcondition:
  - Output artifacts exist and are readable.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

## TC-VIZ-012
- Identifier: `TC-VIZ-012` (links to `UC-VIZ-012`: class-folder and summary consistency)
- Description: Verify `instantiation_summary.json` counts match number of instance `.ply` files per class folder.
- Precondition:
  - Successful run exists.
- Inputs:
  - Artifacts under test:
    - `<run_dir>\instantiation_summary.json`
    - `<run_dir>\<class_name>\*.ply`
- Expected output:
  - For every class key in summary, count equals number of files matching `*_instance_*.ply` in the class folder.
- Expected postcondition:
  - No mismatch across summary vs files.
- Execution history (log):
  - Date/time:
  - Tester:
  - Pass/Fail:
  - Notes:

---

## Reusable Execution Log Template
Use this for each test execution:

```text
Test Case ID:
Use Case ID:
Date/Time:
Tester:
Environment (OS/Python/GPU):
Git Commit:
Preconditions Met (Y/N):
Input Command / Artifact:
Expected Output:
Actual Output:
Expected Postcondition:
Actual Postcondition:
Status (PASS/FAIL/BLOCKED):
Evidence (paths/screenshots/log snippets):
Notes/Defects:
```
