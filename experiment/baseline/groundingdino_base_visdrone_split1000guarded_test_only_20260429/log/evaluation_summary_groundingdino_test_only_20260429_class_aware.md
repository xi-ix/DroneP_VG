# GroundingDINO Test-Only Baseline Summary

## Config
- Dataset: /home/wangzhe/DroneP_VG/dataset/VisDroneSplit1000Guarded
- Split: test only
- Checkpoint: /home/wangzhe/GroundingDINO/weights/groundingdino_swint_ogc.pth
- Config: /home/wangzhe/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
- box_threshold=0.20
- text_threshold=0.20
- nms_threshold=0.40
- eval_mode=class-aware

## Metrics
- Label files considered: 150
- GT boxes: 7973
- Prediction boxes: 5801
- Acc@0.5: 0.3607 (2876/7973)
- Acc@0.75: 0.2779 (2216/7973)
- mAP@0.5: 0.1255
- elapsed_sec: 43.06
