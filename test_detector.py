import subprocess
from typing import List
from os import listdir
from os.path import isdir

data_file = "data/obj.data"
cfg_file = "cfg/yolov3_ndds_final.cfg"
weights_file = "backup/yolov3_ndds_final.weights"

tests_folder = "tests"

tests_paths = [f"{tests_folder}/{f}" for f in listdir(tests_folder) if isdir(f"{tests_folder}/{f}")]

print("test paths: ", tests_paths)

class ObjectData:
    id: int
    left: int
    top: int
    right: int
    bottom: int

    def __init__(self, params):
        try:
            self.id = int(params[0])
            self.left = int(params[1])
            self.top = int(params[2])
            self.right = int(params[3])
            self.bottom = int(params[4])
        except ValueError:
            print("Invalid object data format!")


class Test:
    path_to_test_files: str
    txt_files: List[str]
    img_files: List[str]
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
            std_out = subprocess.Popen(["darknet", "detector", "test", data_file, cfg_file, weights_file, f"{self.path_to_test_files}/{img_file}"], stdout=subprocess.PIPE).communicate()[0]
            predicted_objects = self.load_predictions(std_out.decode("utf-8"))
            self.rate_predictions(real_objects, predicted_objects)
            
    def load_predictions(self, std_out):
        _split = std_out.split("milli-seconds.\r\n")
        print(_split)
        detections = _split[1]
        detections = detections.split("\\n")
        predicted_objects = []
        for detection in detections:
            predicted_object = ObjectData(detection.split(" "))
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



    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def print_test_files(self):
        for img_file, _real in zip(self.img_files, self.real):
            print("Img file: ", img_file)
            for object_data in _real:
                print(object_data.id, object_data.left, object_data.top, object_data.right, object_data.bottom)
                print()

    def print_results(self):
        print(self.path_to_test_files)
        print("avg iou: ", self.iou_sum / self.TP)
        print("TP: ", self.TP)
        print("FN: ", self.FN)
        print("FP: ", self.FP)

if __name__ == "__main__":

    tests = []
    for test_path in tests_paths:
        test = Test(test_path)
        if(test.load_test_files()):
            tests.append(test)

    # debugging
    for test in tests:
        test.print_test_files()

    for test in tests:
        test.run()
    
    for test in tests:
        test.print_results()