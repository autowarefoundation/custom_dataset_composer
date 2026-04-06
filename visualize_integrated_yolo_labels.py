import yaml
import cv2
import numpy as np
import sys

from src.utils.utils import Utils


if __name__ == '__main__':
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    if len(sys.argv) == 2:

        name_dataset = sys.argv[1]

        if name_dataset in config["datasets"]:
            utils = Utils()

            dataset_labels = utils.read_labels_of_dataset_from_txt_folder(
                'datasets/' + name_dataset + '/yolo_labels')

            for dict_labels in dataset_labels:
                path_label_file = list(dict_labels.keys())[0]
                path_img_file = path_label_file.replace(".txt", ".jpg").replace("yolo_labels", "images")
                list_yolo_labels = dict_labels[path_label_file]
                print("[INFO] Image: ", path_img_file)
                img = cv2.imread(str(path_img_file), 1)
                img = np.array(img)

                utils.show_img_with_yolo_labels(img, list_yolo_labels)
                cv2.waitKey(1000)

        else:
            print("[ERROR] Dataset not found in the config file or dataset is not integrated.")

    else:
        print("[ERROR] Wrong number of arguments.")
