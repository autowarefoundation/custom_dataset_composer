import shutil
import time

from .datasets.bdd100k import *
from .datasets.waymo import *
from .datasets.kitti import *
from ..utils.utils import *


class DatasetIntegrator(object):
    def __init__(self, config):
        self.dataset_handlers = None
        self.list_dataset_ready_to_integrate = ['waymo',
                                                'bdd100k',
                                                'kitti']
        self.utils = Utils()
        self.waymo_handler = Waymo(config)
        self.bdd100k_handler = BDD100K(config)
        self.kitti_handler = KITTI(config)

        path_datasets = os.getcwd() + '/datasets'

        if os.path.exists(path_datasets):
            shutil.rmtree(path_datasets)
            print("[INFO] Datasets folder removed.")

        # Iterate over dataset names in cfg file:
        for name_dataset in config["datasets"]:
            # If dataset is in the list, export image and label files:
            if name_dataset in self.list_dataset_ready_to_integrate:

                if name_dataset == "waymo":
                    print("\n[INFO] Exporting image and label files of Waymo Open Dataset ...")
                    start = time.time()
                    os.makedirs(path_datasets + '/waymo/images')
                    os.makedirs(path_datasets + '/waymo/yolo_labels')
                    self.waymo_handler.export_image_and_label_files()
                    dataset_labels = self.utils.read_labels_of_dataset_from_txt_folder(
                        path_datasets + '/waymo/yolo_labels')
                    dict_class_id_label_file_path_waymo = self.utils.analyze_dataset(dataset_labels)
                    self.utils.show_histogram(dict_class_id_label_file_path_waymo, 'Waymo Open Dataset')
                    end = time.time()
                    print("[INFO] Waymo Open Dataset exported in {} seconds.".format(round(end - start), 2))

                if name_dataset == "bdd100k":
                    print("\n[INFO] Exporting image and label files of BDD100K Dataset ...")
                    start = time.time()
                    os.makedirs(path_datasets + '/bdd100k/images')
                    os.makedirs(path_datasets + '/bdd100k/yolo_labels')
                    self.bdd100k_handler.export_image_and_label_files()
                    dataset_labels = self.utils.read_labels_of_dataset_from_txt_folder(
                        path_datasets + '/bdd100k/yolo_labels')
                    dict_class_id_label_file_path_bdd100k = self.utils.analyze_dataset(dataset_labels)
                    end = time.time()
                    self.utils.show_histogram(dict_class_id_label_file_path_bdd100k, 'BDD100K Dataset')
                    print("[INFO] BDD100K dataset exported in {} seconds.".format(round(end - start), 2))

                if name_dataset == "kitti":
                    print("\n[INFO] Exporting image and label files of KITTI Dataset ...")
                    start = time.time()
                    os.makedirs(path_datasets + '/kitti/images')
                    os.makedirs(path_datasets + '/kitti/yolo_labels')
                    self.kitti_handler.export_image_and_label_files()
                    dataset_labels = self.utils.read_labels_of_dataset_from_txt_folder(
                        path_datasets + '/kitti/yolo_labels')
                    dict_class_id_label_file_path_kitti = self.utils.analyze_dataset(dataset_labels)
                    end = time.time()
                    self.utils.show_histogram(dict_class_id_label_file_path_kitti, 'KITTI Dataset')
                    print("[INFO] KITTI Dataset exported in {} seconds.".format(round(end - start), 2))
            else:
                print(name_dataset, "is not in the list of datasets integrated.")
