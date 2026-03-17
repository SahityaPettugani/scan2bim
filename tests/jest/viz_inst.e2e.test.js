const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

jest.setTimeout(30 * 60 * 1000);

const PYTHON_BIN = process.env.VIZ_INST_PYTHON || "python";
const INPUT_FILE =
  process.env.VIZ_INST_INPUT_FILE || "C:\\Users\\iamsa\\Downloads\\synth_room_100.ply";
const CHECKPOINTS_ENV =
  process.env.VIZ_INST_CHECKPOINTS || "C:\\Users\\iamsa\\Downloads\\val_best_miou (1).pth";
const CHECKPOINTS = CHECKPOINTS_ENV.split(";").map((s) => s.trim()).filter(Boolean);
const OUTPUT_ROOT = process.env.VIZ_INST_OUTPUT_DIR || "output_instances";
const RUN_E2E = process.env.RUN_VIZ_INST_E2E === "1";

const REPO_ROOT = path.resolve(__dirname, "..", "..");

function runVizInst(extraArgs = []) {
  const args = ["viz_inst.py", "--input_file", INPUT_FILE];

  for (const ckpt of CHECKPOINTS) {
    args.push("--checkpoint", ckpt);
  }

  args.push("--output_dir", OUTPUT_ROOT);
  args.push("--no-vis-instances");
  args.push(...extraArgs);

  const result = spawnSync(PYTHON_BIN, args, {
    cwd: REPO_ROOT,
    encoding: "utf-8",
  });

  const stdout = result.stdout || "";
  const stderr = result.stderr || "";
  return {
    code: result.status,
    stdout,
    stderr,
    allOutput: `${stdout}\n${stderr}`,
  };
}

function extractRunDirFromOutput(output) {
  const m = output.match(/Run output directory:\s*(.+)/);
  return m ? m[1].trim() : null;
}

function ensureFileExists(p) {
  expect(fs.existsSync(p)).toBe(true);
}

function countInstancePlyFiles(runDir, className) {
  const classDir = path.join(runDir, className);
  if (!fs.existsSync(classDir)) {
    return 0;
  }
  return fs
    .readdirSync(classDir)
    .filter((f) => /^.+_instance_\d{3}\.ply$/i.test(f)).length;
}

const describeIf = RUN_E2E ? describe : describe.skip;

describeIf("viz_inst.py E2E tests mapped to TC-VIZ cases", () => {
  let baselineRunDir = null;

  test("TC-VIZ-001: valid input/checkpoint completes and creates core artifacts", () => {
    const result = runVizInst();

    expect(result.code).toBe(0);
    expect(result.allOutput).toContain("Point Cloud Instantiation Workflow (BIMNet + DBSCAN)");
    expect(result.allOutput).toContain("Step 3: Extracting BIM Parameters and Saving...");

    const runDir = extractRunDirFromOutput(result.allOutput);
    expect(runDir).toBeTruthy();
    baselineRunDir = runDir;

    ensureFileExists(path.join(runDir, "instantiation_summary.json"));
    ensureFileExists(path.join(runDir, "all_instances_combined.ply"));
    ensureFileExists(path.join(runDir, "bim_reconstruction_data.json"));
  });

  test("TC-VIZ-004: missing checkpoint path fails", () => {
    const args = [
      "--checkpoint",
      "C:\\does_not_exist\\bad_checkpoint.pth",
      "--output_dir",
      OUTPUT_ROOT,
      "--no-vis-instances",
    ];
    const cmdArgs = [
      "viz_inst.py",
      "--input_file",
      INPUT_FILE,
      ...args,
    ];
    const result = spawnSync(PYTHON_BIN, cmdArgs, {
      cwd: REPO_ROOT,
      encoding: "utf-8",
    });

    const combined = `${result.stdout || ""}\n${result.stderr || ""}`;
    expect(result.status).not.toBe(0);
    expect(combined.toLowerCase()).toMatch(/(no such file|cannot find|not found|errno)/);
  });

  test("TC-VIZ-006: --no-smooth disables smoothing step", () => {
    const result = runVizInst(["--no-smooth"]);
    expect(result.code).toBe(0);
    expect(result.allOutput).toContain("Step 0.5: KNN smoothing disabled (--no-smooth).");
  });

  test("TC-VIZ-007: smooth auto-skip when point count exceeds threshold", () => {
    const result = runVizInst(["--smooth-max-points", "10"]);
    expect(result.code).toBe(0);
    expect(result.allOutput).toContain("Skipping KNN smoothing because point count");
    expect(result.allOutput).toContain("--smooth-max-points (10)");
  });

  test("TC-VIZ-009: each run gets a unique timestamped output subdirectory", () => {
    const r1 = runVizInst(["--no-smooth"]);
    const r2 = runVizInst(["--no-smooth"]);

    expect(r1.code).toBe(0);
    expect(r2.code).toBe(0);

    const d1 = extractRunDirFromOutput(r1.allOutput);
    const d2 = extractRunDirFromOutput(r2.allOutput);
    expect(d1).toBeTruthy();
    expect(d2).toBeTruthy();
    expect(d1).not.toBe(d2);
    expect(fs.existsSync(d1)).toBe(true);
    expect(fs.existsSync(d2)).toBe(true);
  });

  test("TC-VIZ-010: BIM reconstruction JSON schema sanity", () => {
    if (!baselineRunDir) {
      const result = runVizInst();
      expect(result.code).toBe(0);
      baselineRunDir = extractRunDirFromOutput(result.allOutput);
    }

    const jsonPath = path.join(baselineRunDir, "bim_reconstruction_data.json");
    ensureFileExists(jsonPath);
    const data = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

    expect(Array.isArray(data)).toBe(true);

    for (const obj of data) {
      expect(obj).toHaveProperty("id");
      expect(obj).toHaveProperty("type");
      expect(obj).toHaveProperty("height");
      expect(obj).toHaveProperty("thickness");
      expect(obj).toHaveProperty("geometry");
      expect(typeof obj.height).toBe("number");
      expect(typeof obj.thickness).toBe("number");

      const g = obj.geometry;
      for (const key of ["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]) {
        expect(g).toHaveProperty(key);
        expect(typeof g[key]).toBe("number");
      }
    }
  });

  test("TC-VIZ-012: summary JSON counts match class-folder instance .ply counts", () => {
    if (!baselineRunDir) {
      const result = runVizInst();
      expect(result.code).toBe(0);
      baselineRunDir = extractRunDirFromOutput(result.allOutput);
    }

    const summaryPath = path.join(baselineRunDir, "instantiation_summary.json");
    ensureFileExists(summaryPath);

    const summary = JSON.parse(fs.readFileSync(summaryPath, "utf-8"));
    for (const [className, expectedCount] of Object.entries(summary)) {
      const actualCount = countInstancePlyFiles(baselineRunDir, className);
      expect(actualCount).toBe(expectedCount);
    }
  });
});

describe("Configuration guard", () => {
  test("Set RUN_VIZ_INST_E2E=1 to execute heavy E2E tests", () => {
    if (!RUN_E2E) {
      expect(RUN_E2E).toBe(false);
    } else {
      expect(RUN_E2E).toBe(true);
    }
  });
});
