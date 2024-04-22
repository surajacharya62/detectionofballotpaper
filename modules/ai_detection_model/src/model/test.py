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

    def compare_labels_with_bboxes(self, predicted_labels, true_labels, predicted_bboxes, true_bboxes, scores, image_name, label_id, iou_threshold=0.5):
        matches1 = []
        matches2 = []
        matches3 = []  
        matches4 = [] 
        inv_label = {v:k for k,v in label_id.items()}  
        for pred_label, pred_box,score in zip(predicted_labels, predicted_bboxes, scores):
            best_iou = 0
            value_checked = []
            best_match = {'true_label': None, 'iou': 0, 'valid': False}
            
            
            if pred_label in true_labels:
                extracted_labels = [(true_label, box) for true_label, box in zip(true_labels, true_bboxes) if true_label == pred_label]
                # print(len(extracted_labels))
                if len(extracted_labels) > 1: 
                    for true_label, true_box in extracted_labels:
                        # print(pred_box, true_box)
                        iou = self.calculate_iou(pred_box, true_box)
                           
                        
                        true_label = inv_label.get(true_label,'unkown')
                        pred_label = inv_label.get(pred_label,'unkown')
                        # if iou >= iou_threshold :
                            # best_iou = iou
                        best_match = {'image_name':image_name,
                                       'true_label': true_label,
                                       'true_index': [ true_bboxes.index(true_box) if true_box in true_bboxes else "no index found" ] , 
                                         'pred_index': [predicted_labels.index(pred_label) if pred_label in predicted_labels else 'no index found'],
                                         'class': pred_label,
                                         'Confidence': score,
                                           'iou': iou, 
                                           'TP':  1 if iou > 0.5 else 0, 
                                           'FP': 1 if iou < 0.5 else 0,
                                           'FN':0}
                        value_checked.append(best_match)
                        # print(value_checked)
                    

                    if value_checked:                
                        matches_with_positive_iou = [match for match in value_checked if match['iou'] > 0]                   
                        if matches_with_positive_iou:
                            matches1.extend(matches_with_positive_iou)
                        else:
                            # If no matches with positive IoU, optionally add a single match with iou == 0
                            # Example: Add the first match or define another criterion
                            # matches1.append(value_checked[0])
                            unique_match_found = False
                            for match in value_checked:
                                if match['true_index'] not in [m['true_index'] for m in matches1]:
                                    matches1.append(match)
                                    unique_match_found = True
                                    # break  # Stop after finding the first unique match
                            
                            # if not unique_match_found and value_checked:
                            #     # Optionally, add the first entry if no unique match is found
                            #     # This is a fallback and might not be necessary depending on your requirements
                            #     matches1.append(value_checked[0])
                    
                else:
                    for true_label, true_box in extracted_labels:
                        # print(pred_box, true_box)
                        iou = self.calculate_iou(pred_box, true_box)
                        # print(true_label,true_box)
                        
                        # if iou >= iou_threshold :
                            # best_iou = iou
                            
                        
                        true_label = inv_label.get(true_label,'unkown')
                        pred_label = inv_label.get(pred_label,'unkown')
                        best_match = {'image_name':image_name,
                                      'true_label': true_label,
                                      'true_index': [ true_bboxes.index(true_box) if true_box in true_bboxes else "no index found" ] , 
                                      'class':pred_label,
                                      'Confidence': score,
                                      'pred_index': [predicted_labels.index(pred_label) if pred_label in predicted_labels else 'no index found'], 
                                      'iou': iou, 
                                      'TP': 1 if iou > 0.5 else 0, 
                                      'FP': 1 if iou < 0.5 else 0,
                                      'FN':0}
                    
                        matches2.append(best_match)

            else:
                true_label = inv_label.get(true_label,'unkown')
                pred_label = inv_label.get(pred_label,'unkown')
                best_match = {'image_name':image_name,
                              'true_label': 'nan', 
                              'true_index': 'nan' ,
                               'class':pred_label,
                               'Confidence': score,
                               'pred_index': [predicted_labels.index(pred_label) if pred_label in predicted_labels else 'no index found'],
                                 'iou': 0,
                                 'TP':0, 
                                 'FP': 1,
                                 'FN':0}
                    
                matches3.append(best_match)

                # print

        non_object = set(true_labels) - set(predicted_labels)

        for object in non_object:

            true_label = inv_label.get(true_label,'unkown')
            pred_label = inv_label.get(object,'unkown')
            best_match = {'image_name':image_name,
                            'true_label': 'nan', 
                            'true_index': 'nan' ,
                            'class':pred_label,
                            'Confidence': 0,
                            'pred_index': 'nan',
                                'iou': 0,
                                'TP':0, 
                                'FP': 0,
                                'FN':1}
                
            matches4.append(best_match)

        
        return matches1, matches2, matches3, matches4

# # Example usage
# # predicted_labels = [20, 31]
# # true_labels = [20, 30]

# # predicted_bboxes = [[10, 10, 50, 50], [20, 20, 60, 60]]  # Format: [x1, y1, x2, y2]
# # true_bboxes = [[15, 15, 55, 55], [25, 25, 65, 65]]

# predicted_labels = [15, 13,  5,  5, 20, 14, 29, 20, 28, 39, 41, 41, 24, 42,  1, 34, 19, 32,
#         35, 17, 34, 32, 32, 25, 42,  1, 36,  8,  6,  8, 26, 37, 25, 40, 29, 33,
#         29, 25, 21, 23,  2, 19]
# true_labels = [28, 23, 17, 25, 21, 24, 23, 42, 20, 32, 15, 13, 25,  2, 37, 32, 29,  1,
#         14, 35,  6, 12, 19, 35, 17, 34, 39,  2,  5, 26, 14, 29, 34, 37, 20, 29,
#         36, 41, 22,  5, 41, 19, 30]

# predicted_bboxes = [[1741.8683, 1776.8873, 1927.4122, 1926.2679],
#         [2121.9651, 1764.1221, 2312.6865, 1919.7842],
#         [1737.7671, 2329.4678, 1927.2339, 2484.5891],
#         [1364.0033, 2713.1838, 1544.8967, 2866.2117],
#         [1740.1046, 2526.1919, 1935.7391, 2679.1797],
#         [ 226.0345, 2147.5447,  415.3085, 2295.5696],
#         [2123.1970, 2525.3835, 2306.9609, 2680.0476],
#         [ 984.9064, 1768.2930, 1176.4812, 1918.0055],
#         [ 228.9326, 1578.0225,  417.6078, 1730.4572],
#         [ 986.0679, 2334.7393, 1174.0839, 2484.9360],
#         [1741.7905, 2715.6299, 1930.1272, 2859.8372],
#         [ 605.5679, 2712.1472,  792.3783, 2864.3730],
#         [2117.1648, 1582.8854, 2303.9685, 1727.2869],
#         [ 603.9976, 1768.4097,  795.3846, 1919.0955],
#         [2126.1785, 1948.6403, 2300.5417, 2106.2478],
#         [ 983.5098, 2525.9563, 1175.1162, 2675.8394],
#         [1745.1350, 2151.3311, 1935.2068, 2296.9551],
#         [1359.8551, 1771.1188, 1550.5498, 1919.0995],
#         [ 608.6517, 2147.4573,  803.5634, 2298.6648],
#         [ 985.0970, 1584.1887, 1171.1813, 1729.0994],
#         [ 612.5093, 2340.4219,  796.7136, 2488.9336],
#         [1361.3049, 1958.7073, 1552.4812, 2103.9885],
#         [1360.0422, 2140.0449, 1552.6970, 2295.3408],
#         [ 230.2163, 1959.8611,  415.4686, 2106.6641],
#         [ 231.4475, 2333.6272,  422.1381, 2482.5547],
#         [ 232.6884, 2520.6987,  415.4825, 2668.9116],
#         [ 234.8396, 2713.2612,  415.6931, 2872.5291],
#         [2128.6321, 2146.8452, 2306.3770, 2298.1614],
#         [ 990.7601, 2715.1755, 1174.8783, 2859.1306],
#         [ 609.9813, 1955.9154,  792.9372, 2105.2075],
#         [2116.4497, 2337.1086, 2313.3926, 2486.4700],
#         [2117.6096, 2712.6660, 2313.7302, 2862.8044],
#         [1742.8850, 1586.9409, 1930.5414, 1731.0399],
#         [ 228.0908, 1774.1013,  416.1783, 1914.5491],
#         [ 603.0210, 2531.1277,  796.0768, 2672.0564],
#         [ 981.9566, 1962.7025, 1181.3318, 2104.0632],
#         [1742.6692, 1961.8027, 1934.6727, 2107.6111],
#         [ 995.6221, 2146.4143, 1180.3810, 2294.8118],
#         [1357.7737, 1579.2455, 1548.7361, 1722.2889],
#         [ 609.3972, 1573.6178,  800.0624, 1731.6934],
#         [1361.2094, 2341.0576, 1556.0248, 2483.8396],
#         [1360.8202, 2521.5620, 1566.1335, 2670.2874]]

# true_bboxes = [[ 230., 1580.,  419., 1729.],
#         [ 608., 1580.,  797., 1729.],
#         [ 986., 1580., 1175., 1729.],
#         [1364., 1580., 1553., 1729.],
#         [1742., 1580., 1931., 1729.],
#         [2120., 1580., 2309., 1729.],
#         [ 230., 1769.,  419., 1918.],
#         [ 608., 1769.,  797., 1918.],
#         [ 986., 1769., 1175., 1918.],
#         [1364., 1769., 1553., 1918.],
#         [1742., 1769., 1931., 1918.],
#         [2120., 1769., 2309., 1918.],
#         [ 230., 1958.,  419., 2107.],
#         [ 608., 1958.,  797., 2107.],
#         [ 986., 1958., 1175., 2107.],
#         [1364., 1958., 1553., 2107.],
#         [1742., 1958., 1931., 2107.],
#         [2120., 1958., 2309., 2107.],
#         [ 230., 2147.,  419., 2296.],
#         [ 608., 2147.,  797., 2296.],
#         [ 986., 2147., 1175., 2296.],
#         [1364., 2147., 1553., 2296.],
#         [1742., 2147., 1931., 2296.],
#         [2120., 2147., 2309., 2296.],
#         [ 230., 2336.,  419., 2485.],
#         [ 608., 2336.,  797., 2485.],
#         [ 986., 2336., 1175., 2485.],
#         [1364., 2336., 1553., 2485.],
#         [1742., 2336., 1931., 2485.],
#         [2120., 2336., 2309., 2485.],
#         [ 230., 2525.,  419., 2674.],
#         [ 608., 2525.,  797., 2674.],
#         [ 986., 2525., 1175., 2674.],
#         [1364., 2525., 1553., 2674.],
#         [1742., 2525., 1931., 2674.],
#         [2120., 2525., 2309., 2674.],
#         [ 230., 2714.,  419., 2863.],
#         [ 608., 2714.,  797., 2863.],
#         [ 986., 2714., 1175., 2863.],
#         [1364., 2714., 1553., 2863.],
#         [1742., 2714., 1931., 2863.],
#         [2120., 2714., 2309., 2863.],
#         [1081., 2162., 1193., 2285.]]

# # matches1,matches2,matches3 = compare_labels_with_bboxes(predicted_labels, true_labels, predicted_bboxes, true_bboxes)

# # print(matches)

# # df3 = pd.

# # df = pd.concat(df1[df1['iou']>0],df2)
# # print(df)
# # print(df1[df1['iou']>0])
# # print(df2)
# # # df = matches1 + matches2 + matches3
# # sorted_matches1 = sorted(matches1, key=lambda x: x['true_index'])
# # sorted_matches2 = sorted(matches1, key=lambda x: x['true_index'])
# # df = sorted(df, key=lambda x: x['true_index'])
# # # for match in sorted_matches1:
# # #     print(match)

# # for match in matches3:
# #     print(match)
    


# # Extract elements from list2 where the corresponding element in list1 is in filter_list
# # extracted_elements = [(elem,check) for elem, check in zip(list2, list1) if check in filter_list]
# # extracted_elements = [(true_label, box, true_labels.index(true_label)) for true_label, box in zip(true_labels, true_bboxes) if true_label == 20]
# # print(extracted_elements)
# # for data,value in extracted_elements:
# #     print(data,value)
# #     true_labels.index(data) 

# import pandas as pd

# df = pd.read_excel('df_total_comparisions.xlsx')
# data = pd.DataFrame(df)

# # print(data[0])
# for val in data[0:]:   
#     print(val)