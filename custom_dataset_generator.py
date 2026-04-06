import yaml
import time

from src.custom_dataset_generation.custom_dataset_generator import *
from src.custom_dataset_generation.custom_dataset import *

if __name__ == '__main__':
    if not os.path.exists("datasets"):
        print("There is no datasets folder. Please create one using dataset integrator.")
    else:
        with open('config.yaml') as f:
            config = yaml.safe_load(f)

        utils = Utils()
        train_set_ratio = config['train_set_ratio']
        custom_dataset = CustomDataset()

        start = time.time()

        path_datasets = os.getcwd() + '/datasets/'

        for name_dataset in config['datasets']:
            if name_dataset not in os.listdir("datasets"):
                print("There is no dataset named {}. Please create one using dataset integrator.".format(name_dataset))
            else:
                print("\nGenerating dataset: {}".format(name_dataset))

                dataset_labels = utils.read_labels_of_dataset_from_txt_folder(
                    path_datasets + str(name_dataset) + '/yolo_labels')
                dict_class_id_label_file_path = utils.analyze_dataset(dataset_labels)

                if name_dataset == 'bdd100k':
                    # If the image contains any of the following classes, it will be ignored.
                    # It can be empty list.
                    list_universal_ids_to_ignore = [6, 9, 11]
                    dataset_generator = CustomDatasetGenerator(name_dataset,
                                                               dict_class_id_label_file_path,
                                                               train_set_ratio,
                                                               list_universal_ids_to_ignore,
                                                               config)
                    # Priority is important and if the sample
                    # is added before to custom dataset, don't take
                    dataset_generator.add_samples(1, 9999999)
                    dataset_generator.add_samples(0, 0)
                    dataset_generator.create_label_files()
                    dataset = dataset_generator.split_train_validation()
                    custom_dataset.push_dataset(dataset)

                if name_dataset == 'waymo':
                    # If the image contains any of the following classes, it will be ignored.
                    # It can be empty list.
                    list_universal_ids_to_ignore = []
                    dataset_generator = CustomDatasetGenerator(name_dataset,
                                                               dict_class_id_label_file_path,
                                                               train_set_ratio,
                                                               list_universal_ids_to_ignore,
                                                               config)
                    # Priority is important and if the sample
                    # is added before to custom dataset, don't take
                    dataset_generator.add_samples(2, 9999999)
                    dataset_generator.add_samples(1, 0)
                    dataset_generator.add_samples(0, 0)
                    dataset_generator.create_label_files()
                    dataset = dataset_generator.split_train_validation()
                    custom_dataset.push_dataset(dataset)

                if name_dataset == 'kitti':
                    # If the image contains any of the following classes, it will be ignored.
                    # It can be empty list.
                    list_universal_ids_to_ignore = [12, 16]
                    dataset_generator = CustomDatasetGenerator(name_dataset,
                                                               dict_class_id_label_file_path,
                                                               train_set_ratio,
                                                               list_universal_ids_to_ignore,
                                                               config)
                    # Priority is important and if the sample
                    # is added before to custom dataset, don't take
                    dataset_generator.add_samples(2, 9999999)
                    dataset_generator.add_samples(1, 0)
                    dataset_generator.add_samples(0, 0)
                    dataset_generator.create_label_files()
                    dataset = dataset_generator.split_train_validation()
                    custom_dataset.push_dataset(dataset)

        custom_dataset.analyze()
        custom_dataset.generate_train_valid_files()

        end = time.time()
        print("All process took {} seconds.".format(round(end - start, 2)))

        utils.show_histogram(custom_dataset.dict_class_id_label_file_path_train, "Training Set")
        utils.show_histogram(custom_dataset.dict_class_id_label_file_path_val, "Validation Set")
