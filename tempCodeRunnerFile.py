y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run([detection_boxes, 
																		detection_scores,
																		num_detections,
																		detection_classes], feed_dict=feed_dict)