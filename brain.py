from imageai.Prediction import ImagePrediction
#from imageai.Classification import ImageClassification
import os
execution_path = os.getcwd()

prediction = ImagePrediction()
#prediction = ImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2.h5"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "dog.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)


#https://github.com/OlafenwaMoses/ImageAI