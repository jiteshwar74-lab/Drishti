import os
os.environ['TEMP'] = r'E:\AI_Cache\temp'
os.environ['TMP'] = r'E:\AI_Cache\temp'
os.environ['YOLO_CONFIG_DIR'] = r'E:\AI_Cache\yolo_config'

import torch
# Standard import for modern PyTorch AO (Architecture Optimization)
import torch.ao.quantization as ao_quant
from ultralytics import YOLO

def run_qat():
    # ── 1. Load your trained checkpoint ──────────────────────────────────
    # Using the best weights from your 150-epoch marathon
    model = YOLO(r'E:\AI ML\DRISHTI\models\Medium_1024_Elite\weights\best.pt')
    torch_model = model.model  # Access the raw PyTorch nn.Module
    torch_model.train()

    # ── 2. Fuse layers ───────────────────────────────────────────────────
    # This combines Conv+BN into one layer, making quantization more stable
    torch_model.fuse() 

    # ── 3. Set QAT config ────────────────────────────────────────────────
    # 'qnnpack' is the engine used by Android/iOS (Flutter target)
    torch_model.qconfig = ao_quant.get_default_qat_qconfig('qnnpack')

    # ── 4. Prepare model ─────────────────────────────────────────────────
    # This inserts "FakeQuant" nodes that simulate INT8 precision during training
    ao_quant.prepare_qat(torch_model, inplace=True)

    print("✅ Fake quantization nodes inserted. Model ready for QAT fine-tuning.")
    return model

def qat_finetune():
    model = run_qat()

    # ── 5. Fine-tune with QAT ───────────────────────────────────────────
    results = model.train(
        data=r'E:\AI ML\DRISHTI\datasets\drishti_full_v4\elite_dataset\elite_drishti.yaml',
        epochs=20,             # Keep it short (10-20 epochs is standard)
        imgsz=1024,
        batch=4,               # If this crashes with OOM, drop to 2
        save=True,
        save_period=5,
        lr0=0.0001,            # Very low learning rate so we don't break the weights
        lrf=0.01,
        warmup_epochs=0,
        mosaic=0.5,
        scale=0.3,
        mixup=0.0,
        amp=True,              # Keep AMP enabled for RTX 3060 performance
        label_smoothing=0.05,
        box=7.5,
        cls=2.0,
        device=0,
        workers=2,
        exist_ok=True,
        project='E:/AI ML/DRISHTI/models',
        name='Medium_1024_Elite_QAT',
        close_mosaic=5
    )
    return results

if __name__ == '__main__':
    qat_finetune()