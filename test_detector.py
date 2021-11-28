import subprocess
from os import listdir
import re

data_file = "data/obj.data"
cfg_file = "cfg/yolov3_ndds_final.cfg"
weights_file = "backup/yolov3_ndds_final.weights"
img_file = "final/x.jpg"

easy_tests = "tests/easy_tests"
medium_tests = "tests/medium_tests"
hard_tests = "tests/hard_tests"

tests_paths = [easy_tests, medium_tests, hard_tests]

class ObjectData:
    id: int
    x_center: float
    y_center: float
    x_width: float
    y_width: float

    def __init__(self, params):
        try:
            self.id = int(params[0])
            self.x_center = float(params[1])
            self.y_center = float(params[2])
            self.x_width = float(params[3])
            self.y_width = float(params[4])
        except ValueError:
            print("Invalid object data format!")


class Test:
    path_to_test_files: str
    txt_files: list[str]
    img_files: list[str]
    f1_score: float
    iou_sum: float
    avg_iou: float
    avg_conf: float

    TP: int
    FP: int
    FN: int

    # object data
    real: list[list[ObjectData]]

    def __init__(self, path_to_test_files):
        self.path_to_test_files = path_to_test_files
        self.real = []
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def load_test_files(self):
        img_files = [f for f in listdir(self.path_to_test_files) if f.endswith(".jpg")]
        txt_files = [f"{f.split('.')[0]}.txt" for f in img_files]
        if len(img_files) != len(txt_files):
            print(f"Not enough txt files for {self.path_to_test_files}")
            return False
        self.img_files = img_files
        self.txt_files = txt_files
        self.img_files.sort()
        self.txt_files.sort()

        for txt_file in self.txt_files:
            objects = []
            with open(f"{self.path_to_test_files}/{txt_file}", 'r') as f:
                for line in f:
                    _split = line.split(" ")
                    objects.append(ObjectData(_split))
            self.real.append(objects)

        return True

    
    def run(self):
        for img_file, real_objects in zip(self.img_files, self.real):
            std_out = subprocess.Popen(["darknet", "detector", "test", data_file, cfg_file, weights_file, img_file], stdout=subprocess.PIPE).communicate()[0]
            predicted_objects = self.load_predictions(std_out)
            self.rate_predictions(real_objects, predicted_objects)
            
    def load_predictions(self, std_out):
        _split = std_out.split("milli-seconds.\\r\\n")
        detections = _split[1]
        detections = detections.split("\\n")
        predicted_objects = []
        for detection in detections:
            predicted_object = ObjectData(detection)
            predicted_objects.append(predicted_object)
        return predicted_objects

    def rate_predictions(self, real_objects, predicted_objects):
        for real_object in real_objects:
            for predicted_object in predicted_objects:
                if predicted_object.id == real_object.id:
                    self.TP += 1
                    self.iou_sum += self.calculate_iou(real_object, predicted_object)
                    predicted_objects.remove(predicted_object)
                    break
            self.FN += 1
        self.FP += len(predicted_objects)



    def calculate_iou(self, object_a, object_b):
        boxA = self.change_to_left_top_bottom_right(object_a)
        boxB = self.change_to_left_top_bottom_right(object_b)

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def change_to_left_top_bottom_right(object: ObjectData):
        box = []
        box.append(object.x_center - object.x_width / 2.0)
        box.append(object.y_center - object.y_width / 2.0)
        box.append(object.x_center + object.x_width / 2.0)
        box.append(object.y_center + object.y_width / 2.0)
        return box

    def print_test_files(self):
        for img_file, _real in zip(self.img_files, self.real):
            print("Img file: ", img_file)
            for object_data in _real:
                print(object_data.id, object_data.x_center, object_data.y_center, object_data.x_width, object_data.y_width)
                print()

if __name__ == "__main__":

    tests = []
    for test_path in tests_paths:
        test = Test(test_path)
        if(test.load_test_files()):
            tests.append(test)

    # debugging
    for test in tests:
        test.print_test_files()

    # for test in tests:
    #     test.run()