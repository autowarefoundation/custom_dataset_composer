import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


class Utils(object):
    def __init__(self):
        pass

    # This function takes yolo labels as a dictionary
    # and show image with bounding boxes
    def show_img_with_yolo_labels(self, image, yolo_labels):
        img_height = image.shape[0]
        img_width = image.shape[1]
        # Convert image to numpy array
        image = np.array(image)
        # Convert image to opencv format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Create a copy of image
        image_copy = image.copy()
        # Loop through labels
        for yolo_label in yolo_labels:
            # Get label class
            class_name = yolo_label['class_id']
            bbox_width = img_width * yolo_label["width_norm"]
            bbox_height = img_height * yolo_label["height_norm"]
            bbox_cx = img_width * yolo_label["x_center_norm"]
            bbox_cy = img_height * yolo_label["y_center_norm"]
            cv2.circle(image_copy, (int(bbox_cx), int(bbox_cy)), radius=0,
                       color=(255, 0, 0),
                       thickness=3)

            # Rectangle:
            top_left_corner = (
                int(bbox_cx - (bbox_width / 2)), int(bbox_cy - (bbox_height / 2)))
            # # # represents the bottom right corner of rectangle
            bottom_right_corner = (
                int(bbox_cx + (bbox_width / 2)), int(bbox_cy + (bbox_height / 2)))
            color = (255, 0, 255)
            thickness = 2

            cv2.rectangle(image_copy, bottom_right_corner, top_left_corner, color,
                          thickness)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int(bbox_cx), int(bbox_cy))
            fontScale = 0.4
            fontColor = (255, 255, 255)
            thickness = 1
            lineType = 2

            cv2.putText(image_copy, str((class_name)),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
        cv2.imshow('Sample with YOLO Labels', image_copy)
        cv2.waitKey(500)

    # This function takes path_txt_folder as a parameter.
    # It reads all the txt files containing yolo labels.
    # It returns a list of dictionaries containing txt paths.
    def read_labels_of_dataset_from_txt_folder(self, path_txt_folder):
        list_path_txt_files = glob.glob(str(path_txt_folder) + "/*.txt")
        list_yolo_labels = []
        for path_txt_file in list_path_txt_files:
            with open(path_txt_file, 'r') as f:
                dict_yolo_labels = {}
                dict_yolo_labels[str(path_txt_file)] = []
                lines = f.readlines()
                if len(lines) == 0:
                    print("Empty label file: " + str(path_txt_file))
                for line in lines:
                    line = line.strip()
                    line = line.split(' ')
                    yolo_label = {'class_id': str(line[0]), 'x_center_norm': float(line[1]),
                                  'y_center_norm': float(line[2]), 'width_norm': float(line[3]),
                                  'height_norm': float(line[4])}
                    dict_yolo_labels[str(path_txt_file)].append(yolo_label)
                list_yolo_labels.append(dict_yolo_labels)
        return list_yolo_labels

    def read_labels_of_dataset_from_list_of_path(self, list_path_txt_files):
        list_yolo_labels = []
        for path_txt_file in list_path_txt_files:
            with open(path_txt_file, 'r') as f:
                dict_yolo_labels = {}
                dict_yolo_labels[str(path_txt_file)] = []
                lines = f.readlines()
                if len(lines) == 0:
                    print("Empty label file: " + str(path_txt_file))
                for line in lines:
                    line = line.strip()
                    line = line.split(' ')
                    yolo_label = {'class_id': str(line[0]), 'x_center_norm': float(line[1]),
                                  'y_center_norm': float(line[2]), 'width_norm': float(line[3]),
                                  'height_norm': float(line[4])}
                    dict_yolo_labels[str(path_txt_file)].append(yolo_label)
                list_yolo_labels.append(dict_yolo_labels)
        return list_yolo_labels

    def analyze_dataset(self, list_yolo_labels):
        dict_class_id_label_file_path = {}
        dict_counter_label = {}
        total_label_count = 0
        total_sample_count = 0
        set_class_ids = []
        for sample_labels in list_yolo_labels:
            for label_file_path in sample_labels.keys():
                total_sample_count += 1
                # print("Sample: " + str(key))
                for label in sample_labels[label_file_path]:
                    # print("Label: " + str(label))
                    if label['class_id'] not in set_class_ids:
                        dict_class_id_label_file_path[int(label['class_id'])] = []
                        dict_counter_label[int(label['class_id'])] = 0
                        set_class_ids.append(int(label['class_id']))
                    total_label_count += 1
                    dict_counter_label[int(label['class_id'])] += 1
                    dict_class_id_label_file_path[int(label['class_id'])].append(label_file_path)
                    # print("Label: " + str(label))
                    set_class_ids.append(label['class_id'])
                # print("\n")
            # print("\n")

        print("\nTotal label count: " + str(total_label_count))
        for class_id, label_count in sorted(dict_counter_label.items()):
            print(str(class_id) + " label count: " + str(label_count))

        print("\nTotal sample count: " + str(total_sample_count))
        # Make class ids sorted and set the label paths
        dict_class_id_label_file_path_sorted_set = {}
        for class_id, list_path in sorted(dict_class_id_label_file_path.items()):
            dict_class_id_label_file_path_sorted_set[class_id] = []
            for path in set(list_path):
                dict_class_id_label_file_path_sorted_set[class_id].append(path)
            print(str(len(dict_class_id_label_file_path_sorted_set[class_id])) + " images contain class: " + str(
                class_id))

        return dict_class_id_label_file_path_sorted_set

    def check_file_exist(self, path_train_file):
        if not os.path.isfile(path_train_file):
            # print("File not exist: " + str(path_train_file))
            return False

    def delete_file(self, path_train_file):
        if os.path.isfile(path_train_file):
            os.remove(path_train_file)

    def show_histogram(self, dict, name):
        dist = {}
        for key, values in dict.items():
            dist[key] = len(values)

        c = ['red', 'yellow', 'black', 'blue', 'orange', 'green', 'pink', 'brown', 'purple', 'gray', 'cyan', 'magenta',
             'indigo', 'lime', 'olive', 'teal', 'navy', 'maroon', 'silver', 'gold']

        plt.bar(range(len(dist)), list(dist.values()), align='center', width=0.5, color=c)
        plt.xticks(range(len(dist)), list(dist.keys()))
        plt.xlabel('Universal Classes')
        plt.ylabel('Image Count')
        plt.title('Distribution of classes in {}'.format(name))
        plt.show()
