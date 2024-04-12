from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# Load ground truth COCO data
cocoGt = COCO('true_labels_coco_format.json')

# Load COCO detections, e.g., your model predictions
cocoDt = cocoGt.loadRes('coco_pred.json')

# Create a COCOeval object by passing it the ground truth and detection COCO objects
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox') # 'bbox' for bounding box evaluations

# # Evaluate on a subset of images by setting their ids (optional)
# cocoEval.params.imgIds = [list_of_image_ids] 

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

evalImgs = cocoEval.evalImgs
# print(evalImgs) 


# # Assuming cocoGt is your ground truth COCO object
# gt_image_ids = set(cocoGt.getImgIds())

# # Load prediction JSON to check image IDs
# with open('coco_pred.json') as f:
#     predictions = json.load(f)

# pred_image_ids = {pred['image_id'] for pred in predictions}

# # Find any image IDs in predictions not in ground truth
# unmatched_ids = pred_image_ids - gt_image_ids
# if unmatched_ids:
#     print("Unmatched image IDs:", unmatched_ids)
# else:
#     print("All prediction image IDs match ground truth.")
