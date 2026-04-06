from src.utils.utils import Utils
import glob
import random
import os
from multiprocessing import Pool


class CustomDatasetGenerator(object):
    def __init__(self, dataset_name,
                 dict_class_id_label_file_path,
                 train_set_ratio,
                 set_universal_id_to_remove,
                 config):

        for id_to_remove in set_universal_id_to_remove:
            if id_to_remove not in list(dict_class_id_label_file_path.keys()):
                print("Error: Class id {} not found in dataset.".format(id_to_remove))
                return

        print("Dataset generation started for {} ...".format(dataset_name))
        self.utils = Utils()
        self.config = config
        self.dataset_name = dataset_name
        self.dict_class_id_label_file_path = dict_class_id_label_file_path
        self.train_set_ratio = train_set_ratio
        self.set_universal_id_to_remove = set_universal_id_to_remove
        self.list_class_id_to_add = []
        self.dataset = []
        self.training_set = []
        self.valid_set = []
        self.samples = []

        self.list_path_to_remove = []
        if len(self.set_universal_id_to_remove) > 0:
            for class_id_remove in self.set_universal_id_to_remove:
                # print("Removing class id: {}".format(class_id_remove))
                self.list_path_to_remove += self.dict_class_id_label_file_path[class_id_remove]
        self.list_path_to_remove = list(set(self.list_path_to_remove))
        print("Sample count to be eliminated: {}".format(len(self.list_path_to_remove)),
              "due to unwanted classes: {}\n".format(self.set_universal_id_to_remove))

    def add_or_not(self, target_sample):
        if target_sample in self.list_path_to_remove or target_sample in self.dataset:
            return False
        else:
            return True

    def add_samples(self, class_id, max_sample_count):
        if class_id in list(self.dict_class_id_label_file_path.keys()):
            self.list_class_id_to_add.append(class_id)
            print("Target universal class id:", class_id, " | Max sample count:", max_sample_count)
            target_samples = self.dict_class_id_label_file_path[class_id]
            print("Count images containing target class id:", len(target_samples))
            samples_to_add = []

            if max_sample_count != 0:
                counter = 0

                # Iterate over target samples to determine which samples to add
                # since it takes time long to iterate over all samples, process parallel
                with Pool(int(self.config['count_max_process'])) as process_pool:
                    list_bool_add = process_pool.map(self.add_or_not, target_samples)

                for i, bool_add in enumerate(list_bool_add):
                    if bool_add and counter < max_sample_count:
                        samples_to_add.append(target_samples[i])
                        counter += 1

                self.dataset += samples_to_add
                print("Count images added:", len(samples_to_add))
                print("Count total images added in the dataset: {}\n".format(str(len(self.dataset))))
            else:
                print("Count images added:", len(samples_to_add))
                print("Count total images added in the dataset: {}\n".format(str(len(self.dataset))))
        else:
            print("Class id {} not found in dataset.".format(class_id))
            return

    def create_label_files(self):
        print("Creating label files for {} ...".format(self.dataset_name))

        # Creating label files for the custom dataset:

        # Delete old label files if exist:
        path_img_folder = os.getcwd() + "/datasets/" + str(self.dataset_name) + "/images"
        # Delete all txt files in the path_img_folder:
        for path_txt_file in glob.glob(path_img_folder + "/*.txt"):
            os.remove(path_txt_file)

        for path_label in self.dataset:
            # print("Creating label file for: {}".format(path_label))
            path_new_label_file = path_img_folder + "/" + str(path_label.split('/')[-1])
            path_img_file = path_new_label_file.replace(".txt", ".jpg")
            path_new_label_txt = open(path_new_label_file, "w")
            sample = (self.dataset_name, path_new_label_file, path_img_file)
            with open(path_label, 'r') as f:
                lines = f.readlines()
                is_in = False
                for single_label_line in lines:
                    class_id = int(single_label_line.split(' ')[0])
                    if class_id in self.list_class_id_to_add:
                        row = str(single_label_line.split(' ')[0]) + ' ' \
                              + str(single_label_line.split(' ')[1]) + ' ' \
                              + str(single_label_line.split(' ')[2]) + ' ' \
                              + str(single_label_line.split(' ')[3]) + ' ' \
                              + str(single_label_line.split(' ')[4]).split('\n')[0]
                        path_new_label_txt.write(row + '\n')
                        is_in = True
                if not is_in:
                    print("Warning: Label file {} not found in dataset".format(path_label))
                path_new_label_txt.close()
                self.samples.append(sample)
        print("Custom dataset YOLO label files created into the images folder for {}".format(self.dataset_name))
        print(len(self.samples), "samples created.")


    def split_train_validation(self):
        random.shuffle(self.samples)
        self.training_set = self.samples[:int(len(self.samples) * self.train_set_ratio)]
        self.valid_set = self.samples[int(len(self.samples) * self.train_set_ratio):]
        return self.training_set, self.valid_set
