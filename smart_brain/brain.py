from imageai.Classification import ImageClassification
import os
# Current working directory
exec_path = os.getcwd()

prediction = ImageClassification()
# MobileNetV2 model  (Size = 4.82 mb, fastest prediction time and moderate accuracy)
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(exec_path, 'mobilenet_v2-b0353104.pth'))
prediction.loadModel()

predctions, probabilities = prediction.classifyImage(os.path.join(exec_path,'house.jpg'), result_count=5)
for eachPred, eachProb in zip(predctions, probabilities):
    print(f'{eachPred} : {eachProb}')
    