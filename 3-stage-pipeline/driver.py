import os
import sys
import time
import subprocess

# ── Safety rails ──────────────────────────────────────────────────────────
os.environ['TEMP']            = r'E:\AI_Cache\temp'
os.environ['TMP']             = r'E:\AI_Cache\temp'
os.environ['PIP_CACHE_DIR']   = r'E:\AI_Cache\pip_cache'
os.environ['YOLO_CONFIG_DIR'] = r'E:\AI_Cache\yolo_config'
os.environ['YOLO_OFFLINE']    = 'True'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# ── Change this to wherever your 3 scripts live ───────────────────────────
PIPELINE_DIR = r'E:\AI ML\DRISHTI\3-stage-pipeline'

STAGES = [
    {
        "name"  : "Stage 1 — Full Training (150 epochs)",
        "script": os.path.join(PIPELINE_DIR, "AI_med_train_v4.py"),
        "eta"   : "~18-24 hours"
    },
    {
        "name"  : "Stage 2 — QAT Fine-tune (20 epochs)",
        "script": os.path.join(PIPELINE_DIR, "QAT.py"),
        "eta"   : "~2-3 hours"
    },
    {
        "name"  : "Stage 3 — Export to TFLite int8 (Flutter)",
        "script": os.path.join(PIPELINE_DIR, "Convert_Export.py"),
        "eta"   : "~5-10 minutes"
    },
]

def run_stage(stage: dict, index: int, total: int) -> bool:
    print("\n" + "="*60)
    print(f"  [{index}/{total}] STARTING: {stage['name']}")
    print(f"  ETA    : {stage['eta']}")
    print(f"  Script : {stage['script']}")
    print("="*60 + "\n")

    start  = time.time()
    try:
        # sys.executable is the path to your python.exe
        # stage["script"] is the path to your .py file
        result = subprocess.run([sys.executable, stage["script"]], check=False)
        exit_code = result.returncode
    except Exception as e:
        print(f"System Error: {e}")
        exit_code = 1
    elapsed = (time.time() - start) / 3600  # convert to hours

    if exit_code != 0:
        print("\n" + "❌"*30)
        print(f"  PIPELINE STOPPED — {stage['name']} failed.")
        print(f"  Exit code : {exit_code}")
        print(f"  Action    : Fix the error above, then re-run from this stage.")
        print(f"\n  To resume from this stage, comment out completed")
        print(f"  stages at the bottom of this file using STAGES.pop(0)")
        print("❌"*30)
        return False

    print(f"\n✅ {stage['name']} completed in {elapsed:.2f} hours.")
    return True

def main():
    print("\n" + "🔷"*30)
    print("   DRISHTI — WEAPON DETECTION QAT PIPELINE")
    print("   Flutter / TFLite int8 Target")
    print("🔷"*30)

    # ── Verify all scripts exist before starting ──────────────────────────
    print("\n🔍 Verifying scripts exist...")
    all_found = True
    for stage in STAGES:
        exists = os.path.exists(stage["script"])
        status = "✅" if exists else "❌ NOT FOUND"
        print(f"   {status} — {stage['script']}")
        if not exists:
            all_found = False

    if not all_found:
        print("\n❌ One or more scripts are missing. Fix paths and retry.")
        sys.exit(1)

    print("\n✅ All scripts found. Pipeline is ready.")
    print(f"\nTotal estimated time: ~21-27 hours")
    print("\n⚠️  BEFORE YOU START — checklist:")
    print("   [ ] Laptop/PC plugged in (not on battery)")
    print("   [ ] E:\\AI_Cache\\temp folder exists")
    print("   [ ] Your venv is activated")
    print("   [ ] elite_drishti.yaml has class weights added")
    print("   [ ] best.pt from previous run exists (for Stage 2)")

    input("\nPress ENTER to begin the pipeline, or Ctrl+C to abort... ")

    total = len(STAGES)
    for i, stage in enumerate(STAGES, 1):
        print(f"\n⏭️  Next: {stage['name']} (ETA: {stage['eta']})")
        # input(f"   Press ENTER to start Stage {i}, or Ctrl+C to abort... ")

        success = run_stage(stage, i, total)
        if not success:
            sys.exit(1)

    # ── All done ──────────────────────────────────────────────────────────
    print("\n" + "✅"*30)
    print("   ALL 3 STAGES COMPLETE!")
    print("   Your .tflite int8 model is ready for Flutter.")
    print(f"\n   📁 Find it at:")
    print(f"   E:/AI ML/DRISHTI/models/Medium_1024_Elite_INT8_tflite/")
    print("✅"*30)

# ── RESUMING FROM A SPECIFIC STAGE? ──────────────────────────────────────
# If Stage 1 already finished, comment in the line below to skip it:
STAGES.pop(0)   # ← skips Stage 1
STAGES.pop(0)   # ← skips Stage 2 (use both lines to jump to Stage 3)

if __name__ == '__main__':
    main()