from sklearn.utils import class_weight
from lib.AlexNet import model_generator
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.optimizers import SGD
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import keras.callbacks

def mean_pred(y_true, y_pred):
    return K.mean(y_pred[:, 1]>0.5)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Train based on Zone")
    parser.add_argument('Zone', type=int,help='Positional zone number:int')
    parser.add_argument('literation', type=int,help='Positional training literation number:int')

    args = parser.parse_args()


    sys.stderr.write("Loading Data.....\n")
    Y =np.load("../data/processed/Zone_%d/Y.npy" % args.Zone)
    X =np.load("../data/processed/Zone_%d/X.npy" % args.Zone)

    accuracy_threshold = np.mean(Y[:,0])

    model = model_generator(X[0, :, :, :].shape)

    opt = SGD(lr=0.004)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', mean_pred])

    class_weight_1 = class_weight.compute_class_weight('balanced', np.unique(Y[:, 1]), Y[:, 1])
    class_weight_dict = dict(enumerate(class_weight_1))

    #earlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001,patience=10,verbose=0, mode='auto')

    class Metrics(keras.callbacks.Callback):

        def on_train_begin(self, logs={}):
            self.precision = []
            self.recall = []

        def on_epoch_end(self, batch, logs={}):
            y_predict = np.asarray(self.model.predict(self.validation_data[0])[:,1])
            y_true = np.asarray(self.validation_data[1][:,1])
            self.precision.append(sum(np.round(y_predict)*y_true)/sum(np.round(y_predict)))
            self.recall.append(sum(np.round(y_predict)*y_true)/sum(y_true))
            sys.stderr.write("- val_precision: %.4f - val_recall: %.4f" % (self.precision[-1], self.recall[-1])+"\n")
            return



    metrics = Metrics()

    fit_process = model.fit(X, Y,  # Train the model using the training set...
                            batch_size=20, epochs=args.literation, class_weight=class_weight_dict,
                            verbose=1, validation_split=0.3,shuffle=True,callbacks=[metrics])

    if os.path.exists("../output") == False:
        os.mkdir("../output")

    if os.path.exists("../output/Zone_"+ str(args.Zone)) == False:
        os.mkdir("../output/Zone_"+ str(args.Zone))

    sys.stderr.write("Visualization model evaluation.....\n")

    plt.figure()
    plt.plot(fit_process.history["loss"], color = 'blue',label='Training')
    plt.plot(fit_process.history["val_loss"], color = 'green',label='Testing')
    plt.title("Loss Function")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("../output/Zone_%d/Loss.png" % args.Zone)

    plt.figure()
    plt.plot(fit_process.history["acc"], color = 'blue',label='Training')
    plt.title("Weighted Model Training Accuracy")
    plt.xlabel("Time")
    plt.ylabel("Weighted Accuracy")
    plt.savefig("../output/Zone_%d/Training_Accuracy.png" % args.Zone)

    plt.figure()
    plt.plot(fit_process.history["val_acc"], color = 'blue',label='Validation')
    plt.title(" Model Validation Accuracy")
    plt.xlabel("Time")
    plt.ylabel("Accuracy")
    plt.savefig("../output/Zone_%d/Validation_Accuracy.png" % args.Zone)



    plt.figure()
    plt.plot(metrics.recall, color = 'blue',label='Recall')
    plt.plot(metrics.precision, color = 'green',label='Precision')
    plt.title(" Model Evaluation: Recall and Precision")
    plt.xlabel("Time")
    plt.ylabel("Rate")
    plt.legend()
    plt.savefig("../output/Zone_%d/Recall_Precision.png" % args.Zone)



    sys.stderr.write("Save Model Weight.....\n")

    np.save("../output/Zone_%d/training_log" % args.Zone, fit_process.history)
    model.save_weights("../output/Zone_%d/model_weight" % args.Zone)

    if os.path.exists("../output/Summary.csv") == False:
        with open("../output/Summary.csv",'w') as f:
            f.write("Zone,Threshold,Train_acc,Val_acc,Val_recall,Val_precision\n")

    with open("../output/Summary.csv", 'a') as f:
        f.write("%d, %.4f, %.4f, %.4f, %.4f, %.4f\n" % (args.Zone, accuracy_threshold,fit_process.history["acc"][-1],
                                                     fit_process.history["val_acc"][-1],metrics.recall[-1],metrics.precision[-1]))
