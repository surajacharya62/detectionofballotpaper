# # import pandas as pd

# # listi = [(7,8,9),(3,4,5),(7,8,9)]
# # # listi.append(('','image_name','pred_label','actual_label'))

# # df = pd.DataFrame(listi, columns=['Image_name1', 'Image_name2', 'pred_labels'])

# # print(listi)
# # df[]
# # print(sorted(df['Image_name1']))

# import pandas as pd

# # Assuming df is your DataFrame
# data = {
#     'Image_name1': ['image_0002.jpg', 'image_0002.jpg', 'image_0002.jpg', 'image_0002.jpg'],
#     'Image_name2': ['image_0002.jpg', 'image_0002.jpg', 'image_0002.jpg', 'image_0002.jpg'],
#     'pred_labels': ['tensor(15)', 'tensor(13)', 'tensor(5)', 'tensor(5)'],
#     'actual_labels': ['tensor(28)', 'tensor(23)', 'tensor(17)', 'tensor(25)']
# }
# df = pd.DataFrame(data)

# # Extracting numerical values from the tensor strings
# df['pred_labels_num'] = df['pred_labels'].str.extract('(\d+)').astype(int)
# df['actual_labels_num'] = df['actual_labels'].str.extract('(\d+)').astype(int)

# # Sorting by the numerical columns
# df_sorted = df.sort_values(by=['pred_labels_num', 'actual_labels_num'])

# # If you want to remove the helper columns and keep only the original format
# df_sorted.drop(['pred_labels_num', 'actual_labels_num'], axis=1, inplace=True)

# # print(df_sorted)

# # zip(final_boxes, final_scores, final_labels, actual_labels):

# # from collections import Counter

# # # predicted_labels = [15, 13, 5, 5, 20, 14]  # Example predicted labels
# # # true_labels = [28, 23, 17, 5, 21, 24]     # Example true labels

# # predicted_labels = [15, 13,  5,  5, 20, 14, 29, 20, 28, 39, 41, 41, 24, 42,  1, 34, 19, 32,
# #         35, 17, 34, 32, 32, 25, 42,  1, 36,  8,  6,  8, 26, 37, 25, 40, 29, 33,
# #         29, 25, 21, 23,  2, 19]
# # true_labels = [28, 23, 17, 25, 21, 24, 23, 42, 20, 32, 15, 13, 25,  2, 37, 32, 29,  1,
# #         14, 35,  6, 12, 19, 35, 17, 34, 39,  2,  5, 26, 14, 29, 34, 37, 20, 29,
# #         36, 41, 22,  5, 41, 19, 30]

# # # Step 1: Identify correctly predicted labels
# # correct_labels = set(predicted_labels) & set(true_labels)
# # print(correct_labels)

# # # Step 2: Identify missed and falsely predicted labels
# # missed_labels = set(true_labels) - set(predicted_labels)
# # falsely_predicted_labels = set(predicted_labels) - set(true_labels)

# # # Optional: Detailed comparison using Counter
# # predicted_counts = Counter(predicted_labels)
# # true_counts = Counter(true_labels)

# # # Display results
# # print("Correctly Predicted Labels (regardless of frequency):", correct_labels)
# # print("Missed Labels:", missed_labels)
# # print("Falsely Predicted Labels:", falsely_predicted_labels)

# # # For detailed analysis: Check the counts
# # # This part is for understanding which labels were over/under-predicted
# # print("\nDetailed Comparison:")
# # for label in set(true_labels + predicted_labels):
# #     print(f"Label {label}: True Count = {true_counts[label]}, Predicted Count = {predicted_counts[label]}")
# #----------------------------
# # def compare_unordered_labels(predicted_labels, true_labels):
# #     """
# #     Compares predicted labels against true labels without considering the order.
# #     Ensures each label is matched only once.
    
# #     Args:
# #     - predicted_labels (list): The list of predicted labels.
# #     - true_labels (list): The list of true labels in their correct order.
    
# #     Returns:
# #     - correct_matches (int): Number of correctly matched labels.
# #     - unmatched_predictions (list): Predicted labels that couldn't be matched.
# #     - unmatched_trues (list): True labels that weren't matched by any prediction.
# #     """
# #     true_labels_copy = true_labels.copy()
# #     correct_matches = 0
# #     unmatched_predictions = []

# #     for predicted in predicted_labels:
# #         if predicted in true_labels_copy:
# #             correct_matches += 1
# #             true_labels_copy.remove(predicted)  # Mark off the matched label
# #         else:
# #             unmatched_predictions.append(predicted)

# #     return correct_matches, unmatched_predictions, true_labels_copy

# # # Example usage
# # predicted_labels = [15, 13, 5, 5, 20, 14]  # Random order
# # true_labels = [28, 23, 17, 5, 21, 24]  # Correct order

# # correct_matches, unmatched_predictions, unmatched_trues = compare_unordered_labels(predicted_labels, true_labels)

# # print(f"Correct Matches: {correct_matches}")
# # print(f"Unmatched Predictions: {unmatched_predictions}")
# # print(f"Unmatched True Labels: {unmatched_trues}")
# #------------------------------
# # from collections import Counter

# # def detailed_label_comparison(predicted_labels, true_labels):
# #     predicted_counts = Counter(predicted_labels)
# #     true_counts = Counter(true_labels)
    
# #     correct_matches = 0
# #     for label, pred_count in predicted_counts.items():
# #         true_count = true_counts.get(label, 0)
# #         correct_matches += min(pred_count, true_count)  # Count as correct match the minimum of the two counts

# #     return correct_matches, len(predicted_labels) - correct_matches, len(true_labels) - correct_matches

# # # Example usage
# # # predicted_labels = [15, 13, 5, 5, 20, 14, 5]  # Predicted labels with repetitions
# # # true_labels = [28, 23, 17, 25, 21, 24, 5, 5, 5]  # True labels with repetitions

# # predicted_labels = [15, 13,  5,  5, 20, 14, 29, 20, 28, 39, 41, 41, 24, 42,  1, 34, 19, 32,
# #         35, 17, 34, 32, 32, 25, 42,  1, 36,  8,  6,  8, 26, 37, 25, 40, 29, 33,
# #         29, 25, 21, 23,  2, 19]
# # true_labels = [28, 23, 17, 25, 21, 24, 23, 42, 20, 32, 15, 13, 25,  2, 37, 32, 29,  1,
# #         14, 35,  6, 12, 19, 35, 17, 34, 39,  2,  5, 26, 14, 29, 34, 37, 20, 29,
# #         36, 41, 22,  5, 41, 19, 30]

# # correct_matches, over_predictions, under_predictions = detailed_label_comparison(predicted_labels, true_labels)

# # print(f"Correct Matches: {correct_matches}")
# # print(f"Over Predictions (Predicted not in True): {over_predictions}")
# # print(f"Under Predictions (True not Predicted): {under_predictions}")

# #-------------------------------

# # import pandas as pd

# # def compare_and_record(true_labels, predicted_labels, image_id):
# #     # Copy the predicted labels list to manipulate it without affecting the original
# #     predicted_labels_copy = predicted_labels.copy()
    
# #     # Prepare a list to hold the comparison results
# #     comparison_results = []

# #     for true_label in true_labels:
# #         if true_label in predicted_labels_copy:
# #             # If a match is found, mark it as valid and remove one instance of this label from the predicted list
# #             comparison_results.append({"image_id": image_id, "true_label": true_label, "predicted_label": true_label, "valid": True})
# #             predicted_labels_copy.remove(true_label)  # Remove to prevent double-counting
# #         else:
# #             # If no match is found, mark as invalid
# #             comparison_results.append({"image_id": image_id, "true_label": true_label, "predicted_label": None, "valid": False})
    
# #     # Convert the comparison results to a DataFrame
# #     results_df = pd.DataFrame(comparison_results)
# #     return results_df

# # # Example data
# # image_id = "image_id_001"
# # # true_labels = [20, 30, 20, 15]  # True labels might contain duplicates
# # # predicted_labels = [30, 20, 15, 50]  # Predicted labels are unordered and might not match true labels exactly
# # predicted_labels = [15, 13,  5,  5, 20, 14, 29, 20, 28, 39, 41, 41, 24, 42,  1, 34, 19, 32,
# #         35, 17, 34, 32, 32, 25, 42,  1, 36,  8,  6,  8, 26, 37, 25, 40, 29, 33,
# #         29, 25, 21, 23,  2, 19]
# # true_labels = [28, 23, 17, 25, 21, 24, 23, 42, 20, 32, 15, 13, 25,  2, 37, 32, 29,  1,
# #         14, 35,  6, 12, 19, 35, 17, 34, 39,  2,  5, 26, 14, 29, 34, 37, 20, 29,
# #         36, 41, 22,  5, 41, 19, 30]

# # # Compare and create a DataFrame of results
# # results_df = compare_and_record(true_labels, predicted_labels, image_id)
# # print(results_df)
# #--------------------

# from model.visualize_prediction import VisualizePrediction
# obj_viz = VisualizePrediction()

from visualize.visualize_prediction import VisualizePrediction
import pandas as pd
import torch
obj_viz = VisualizePrediction()

class CompareBoundingBox:

    def labels(self, test_set, predictions, label_id):
        total_comparisions = []
        total_comparisions1 = []
        for i, (test_data, label) in enumerate(zip(test_set, predictions)):  
            # print(test_data, label)

            actual_labels = test_data[3]['labels']  
            true_bboxes = test_data[3]['boxes']
            image_name1 = test_data[2]

            predicted_bboxes = label[0]['boxes'] 
            predicted_labels = label[0]['labels']
            scores = label[0]['scores']
            image_name = label[2]   

            indices = obj_viz.apply_nms(predicted_bboxes, scores)   
            # final_boxes, final_scores, final_labels = self.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )      

            final_boxes, final_scores, final_labels  =  predicted_bboxes[indices], scores[indices], predicted_labels[indices]   
            # print(final_boxes)
            # print(final_labels)
            final_boxes = final_boxes.tolist()
            final_labels = final_labels.tolist()
            final_scores =  final_scores.tolist()
            true_bboxes = true_bboxes.tolist()
            actual_labels = actual_labels.tolist()


            # print(final_boxes)
            # print(final_labels)

            matches1,matches2,matches3,matches4 = self.compare_labels_with_bboxes(final_labels, actual_labels, final_boxes, true_bboxes, final_scores, image_name1,label_id, iou_threshold=0.5)
            df = matches1 + matches2 + matches3 + matches4
            
            total_comparisions.append(df)
            # total_comparisions1.append(matches1)
            # total_comparisions1.append(matches2)
            # total_comparisions1.append(matches3)

        
        data = pd.DataFrame(total_comparisions)
        data.to_excel('../../../faster_rcnn_files/df_total_comparisions.xlsx')
        # data1 = pd.DataFrame(total_comparisions1)
        # data1.to_excel('df_total_comparisions1.xlsx')



    def calculate_iou(self,boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Calculate the area of intersection
        intersection_area = max(0, xB - xA) * max(0, yB - yA)

        # Calculate the areas of both bounding boxes
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Calculate the area of union
        union_area = boxA_area + boxB_area - intersection_area

        # Compute the IoU
        iou = intersection_area / float(union_area)

        return iou

    # def compare_labels_with_bboxes(self, predicted_labels, true_labels, predicted_bboxes, true_bboxes, scores, image_name, label_id, iou_threshold=0.5):
    #     matches1 = []
    #     matches2 = []
    #     matches3 = []  
    #     matches4 = [] 
    #     inv_label = {v:k for k,v in label_id.items()}  
    #     for pred_label, pred_box,score in zip(predicted_labels, predicted_bboxes, scores):
    #         best_iou = 0
    #         value_checked = []
    #         best_match = {'true_label': None, 'iou': 0, 'valid': False}
            
            
    #         if pred_label in true_labels:
    #             extracted_labels = [(true_label, box) for true_label, box in zip(true_labels, true_bboxes) if true_label == pred_label]
    #             # print(len(extracted_labels))
    #             if len(extracted_labels) > 1: 
    #                 for true_label, true_box in extracted_labels:
    #                     # print(pred_box, true_box)
    #                     iou = self.calculate_iou(pred_box, true_box)
                           
                        
    #                     true_label = inv_label.get(true_label,'unkown')
    #                     pred_label = inv_label.get(pred_label,'unkown')
    #                     # if iou >= iou_threshold :
    #                         # best_iou = iou
    #                     best_match = {'image_name':image_name,
    #                                    'true_label': true_label,
    #                                    'true_index': [ true_bboxes.index(true_box) if true_box in true_bboxes else "no index found" ] , 
    #                                      'pred_index': [predicted_labels.index(pred_label) if pred_label in predicted_labels else 'no index found'],
    #                                      'class': pred_label,
    #                                      'Confidence': score,
    #                                        'iou': iou, 
    #                                        'TP':  1 if iou > 0.5 else 0, 
    #                                        'FP': 1 if iou < 0.5 else 0,
    #                                        'FN':0}
    #                     value_checked.append(best_match)
    #                     # print(value_checked)
                    

    #                 if value_checked:                
    #                     matches_with_positive_iou = [match for match in value_checked if match['iou'] > 0]                   
    #                     if matches_with_positive_iou:
    #                         matches1.extend(matches_with_positive_iou)
    #                     else:
    #                         # If no matches with positive IoU, optionally add a single match with iou == 0
    #                         # Example: Add the first match or define another criterion
    #                         # matches1.append(value_checked[0])
    #                         unique_match_found = False
    #                         for match in value_checked:
    #                             if match['true_index'] not in [m['true_index'] for m in matches1]:
    #                                 matches1.append(match)
    #                                 unique_match_found = True
    #                                 # break  # Stop after finding the first unique match
                            
    #                         # if not unique_match_found and value_checked:
    #                         #     # Optionally, add the first entry if no unique match is found
    #                         #     # This is a fallback and might not be necessary depending on your requirements
    #                         #     matches1.append(value_checked[0])
                    
    #             else:
    #                 for true_label, true_box in extracted_labels:
    #                     # print(pred_box, true_box)
    #                     iou = self.calculate_iou(pred_box, true_box)
    #                     # print(true_label,true_box)
                        
    #                     # if iou >= iou_threshold :
    #                         # best_iou = iou
                            
                        
    #                     true_label = inv_label.get(true_label,'unkown')
    #                     pred_label = inv_label.get(pred_label,'unkown')
    #                     best_match = {'image_name':image_name,
    #                                   'true_label': true_label,
    #                                   'true_index': [ true_bboxes.index(true_box) if true_box in true_bboxes else "no index found" ] , 
    #                                   'class':pred_label,
    #                                   'Confidence': score,
    #                                   'pred_index': [predicted_labels.index(pred_label) if pred_label in predicted_labels else 'no index found'], 
    #                                   'iou': iou, 
    #                                   'TP': 1 if iou > 0.5 else 0, 
    #                                   'FP': 1 if iou < 0.5 else 0,
    #                                   'FN':0}
                    
    #                     matches2.append(best_match)

    #         else:
    #             true_label = inv_label.get(true_label,'unkown')
    #             pred_label = inv_label.get(pred_label,'unkown')
    #             best_match = {'image_name':image_name,
    #                           'true_label': 'nan', 
    #                           'true_index': 'nan' ,
    #                            'class':pred_label,
    #                            'Confidence': score,
    #                            'pred_index': [predicted_labels.index(pred_label) if pred_label in predicted_labels else 'no index found'],
    #                              'iou': 0,
    #                              'TP':0, 
    #                              'FP': 1,
    #                              'FN':0}
                    
    #             matches3.append(best_match)

    #             # print

    #     non_object = set(true_labels) - set(predicted_labels)

    #     for object in non_object:

    #         true_label = inv_label.get(true_label,'unkown')
    #         pred_label = inv_label.get(object,'unkown')
    #         best_match = {'image_name':image_name,
    #                         'true_label': 'nan', 
    #                         'true_index': 'nan' ,
    #                         'class':pred_label,
    #                         'Confidence': 0,
    #                         'pred_index': 'nan',
    #                             'iou': 0,
    #                             'TP':0, 
    #                             'FP': 0,
    #                             'FN':1}
                
    #         matches4.append(best_match)

        
    #     return matches1, matches2, matches3, matches4


#---------------------------------
    def compare_labels_with_bboxes(self, predicted_labels, true_labels, predicted_bboxes, true_bboxes, scores, image_name,label_id, iou_threshold=0.5):
        matches1 = []
        matches2 = []
        matches3 = []  
        matches4 = [] 
        inv_label = {v:k for k,v in label_id.items()}  
        # print(true_bboxes, predicted_bboxes)
        
        for tlabel, tbox in zip(true_labels, true_bboxes):
            best_iou = 0
            value_checked = []
            
            best_match = {'true_label': None, 'iou': 0, 'valid': False}
            # print(true_labels, true_bboxes)
            # print(true_labels[0])
            # true_labels = true_labels.cpu().numpy()
            # true_bboxes = true_bboxes.cpu().numpy()
            # print(true_labels)
            # tlabel = label_id.get(tlabel, "unkown")
            tlabel = tlabel
            # print(label_id, "tlabel")

            for plabel, pbox, score in zip(predicted_labels, predicted_bboxes, scores):
               
                # if isinstance(tbox, torch.Tensor):
                #     tbox_list = tbox.tolist()  # Convert to list
                # else:
                #     tbox_list = tbox 

                # print("tbox_list:", tbox_list)
                iou = self.calculate_iou(pbox, tbox) 
                # print(plabel)
                # print(inv_label)
                # plabel = label_id.get(plabel, 'unkown')
                # print(plabel,tlabel, 'plabel', pbox,tbox)

                if iou > iou_threshold and tlabel == plabel:

                    
                        # pred_label = inv_label.get(pred_label, 'unkown')
                        # if iou >= iou_threshold :
                            # best_iou = iou
                    best_match = {'image_name':image_name,
                                    'true_label':inv_label.get(tlabel,'unknown'),
                                    'pred_label':inv_label.get(plabel,'unknown'),                                 
                                    'class': inv_label.get(tlabel,'known'),
                                    'Confidence': score,
                                    'iou': iou, 
                                    'TP':  1, 
                                    'FP': 0,
                                    'FN':0}
                    matches1.append(best_match)
                
                elif iou > iou_threshold and tlabel != plabel:

                    best_match = {'image_name':image_name,
                                    'true_label':inv_label.get(tlabel,'unknown'),
                                    'pred_label':inv_label.get(plabel,'unknown'),                                 
                                    'class': inv_label.get(tlabel,'known'),
                                    'Confidence': score,
                                    'iou': iou, 
                                    'TP':  0, 
                                    'FP': 1,
                                    'FN':0}
                    matches2.append(best_match)
                
               
        # pred_labels = [label_id.get(label) for label in predicted_labels]
        
        true_labels1 = true_labels
        # print(true_labels1)

        # p_list = []
        # for p_label, score in pred_labels:
        #     p_list.append(p_label)
        
        non_object_detected = set(true_labels) - set(predicted_labels)
        # print(non_object_detected)
        

        for label in non_object_detected:
            
            pred_label1 = inv_label.get(label)

            # true_label = inv_label.get(true_label,'unkown')
            # pred_label = inv_label.get(object,'unkown')
            best_match = {'image_name':image_name,
                             'true_label': inv_label.get(label, 'unkown'),
                             'pred_label':'Not detected', 
                           
                            'class':inv_label.get(label, 'unknown'),
                            'Confidence': 0,
                            
                                'iou': 0,
                                'TP':0, 
                                'FP': 0,
                                'FN':1}
                
            matches4.append(best_match)

        
        return matches1, matches2, matches3, matches4     
