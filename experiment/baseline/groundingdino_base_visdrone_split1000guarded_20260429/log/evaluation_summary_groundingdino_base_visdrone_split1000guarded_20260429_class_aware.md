# GroundingDINO Baseline Summary

## Config
- Dataset: /home/wangzhe/DroneP_VG/dataset/VisDroneSplit1000Guarded
- Checkpoint: /home/wangzhe/GroundingDINO/weights/groundingdino_swint_ogc.pth
- Config: /home/wangzhe/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
- box_threshold=0.20
- text_threshold=0.20
- nms_threshold=0.40
- eval_mode=class-aware

## Metrics
- Label files considered: 1000
- GT boxes: 53121
- Prediction boxes: 38088
- Acc@0.5: 0.3391 (18014/53121)
- Acc@0.75: 0.2600 (13814/53121)
- mAP@0.5: 0.1099
- elapsed_sec: 208.52
