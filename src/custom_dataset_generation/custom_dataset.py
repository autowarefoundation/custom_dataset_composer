import os
import random

from src.utils.utils import Utils


class CustomDataset(object):
    def __init__(self):
        self.utils = Utils()
        self.list_training_set_combined = []
        self.list_valid_set_combined = []
        self.dict_class_id_label_file_path_train = {}
        self.dict_class_id_label_file_path_val = {}

    def push_dataset(self, dataset):
        training_set, valid_set = dataset
        for sample_train in training_set:
            self.list_training_set_combined.append(sample_train)
        for sample_valid in valid_set:
            self.list_valid_set_combined.append(sample_valid)
        random.shuffle(self.list_training_set_combined)
        random.shuffle(self.list_valid_set_combined)

    def analyze(self):
        print("Analyzing training set...")
        list_label_path_train = [str(label_path) for dataset_name, label_path, img_path in
                                 self.list_training_set_combined]
        list_yolo_labels_train = self.utils.read_labels_of_dataset_from_list_of_path(list_label_path_train)
        self.dict_class_id_label_file_path_train = self.utils.analyze_dataset(list_yolo_labels_train)
        print("Analyzing validation set...")
        list_label_path_val = [str(label_path) for dataset_name, label_path, img_path in
                               self.list_valid_set_combined]
        list_yolo_labels_val = self.utils.read_labels_of_dataset_from_list_of_path(list_label_path_val)
        self.dict_class_id_label_file_path_val = self.utils.analyze_dataset(list_yolo_labels_val)

    def generate_train_valid_files(self):
        # Open train.txt and valid.txt in current directory
        # If train.txt and valid.txt exist, delete them
        path_train_file = os.getcwd() + "/train.txt"
        path_valid_file = os.getcwd() + "/valid.txt"
        if self.utils.check_file_exist(path_train_file):
            self.utils.delete_file(path_train_file)
        if self.utils.check_file_exist(path_valid_file):
            self.utils.delete_file(path_valid_file)
        train_file = open(path_train_file, "w")
        valid_file = open(path_valid_file, "w")
        # Write train.txt and valid.txt
        for dataset_name, label_path, img_path in self.list_training_set_combined:
            train_file.write(str(img_path) + "\n")
        for dataset_name, label_path, img_path in self.list_valid_set_combined:
            valid_file.write(str(img_path) + "\n")
        print("\nGenerated train.txt and valid.txt")

