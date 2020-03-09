# train_logit.py, train_svc.py, predict.py 

1. train_logit.py is a copy of the train.py except that model is saved as logit.cpickle instead of model.cpickle, and it’s saved under the current directory. (Transfer-learning-keras) 
2. train_svc.py, implemented with GridSearchCV() but only hyper parameter ‘C’ is turned as for now. Same with train_logit.py, model is saved under transfer-learning-keras as svc.cpickle. 

Steps: 

1. Run train_logit.py or python train_svc.py in CL

	Result:  2 files(logit.cpickle, svc.cpickle) will be saved under “transfer-learning-keras” directory (parallel to the original train.py). 

2. Run command:    python predict.py image_path model_name  
	for example: 

PastedGraphic-2.png

	Result:  
		Prediction Model used is: svc.cpickle 
		It is food. 

PastedGraphic-4.png
	Result:
	Prediction Model used is: logit.cpickle 
	It is not food. 


3. Known issues:
	a. predict.py is only to predict single image. 
 	b. Warning messages are note muted. Depending on the configuration of running environment, there might be multiple warning messages returned, which can be ignored. 
	c. A message telling what training model is used will be printed using the .cpickle
