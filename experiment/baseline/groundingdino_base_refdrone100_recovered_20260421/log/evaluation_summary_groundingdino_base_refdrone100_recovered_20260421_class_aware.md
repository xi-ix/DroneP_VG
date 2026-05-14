# GroundingDINO Baseline Summary

## Config
- Dataset: /home/wangzhe/DroneP_VG/visdrone_test_100
- Checkpoint: /home/wangzhe/GroundingDINO/weights/groundingdino_swint_ogc.pth
- Config: /home/wangzhe/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
- box_threshold=0.20
- text_threshold=0.20
- nms_threshold=0.40
- eval_mode=class-aware

## Metrics
- Label files considered: 100
- GT boxes: 4304
- Prediction boxes: 3130
- Acc@0.5: 0.3083 (1327/4304)
- Acc@0.75: 0.2407 (1036/4304)
- mAP@0.5: 0.1038
- elapsed_sec: 56.16
