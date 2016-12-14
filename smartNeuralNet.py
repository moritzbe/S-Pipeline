from data_tools import *
from algorithms import *
from plot_lib import *
from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
import numpy as np
import code 

# X
# X_test etc...
#
#


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds, X, X_test, y):
    # input image dimensions
    batch_size = 8
    nb_epoch = 50
    random_state = 51

    train_data = X
    train_target = y

    yfull_train = dict()
    kf = KFold(len(y), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    accuracies = 0
    models = []
    for train_index, test_index in kf:

        ########## !!!!!!!!!!!!! #########
        model = betterFullyConnectedNet(X, y, epochs)
        ########## !!!!!!!!!!!!! #########
        
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=2, validation_data=(X_valid, Y_valid), callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        test_prediction = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        y_pred = np.zeros([test_prediction.shape[0]])
        for i in xrange(test_prediction.shape[0]):
            y_pred[i] = np.argmax(test_prediction[i,:]).astype(int)
        plotConfusionMatrix(Y_valid.astype(int), y_pred.astype(int))

        scores = model.evaluate(X_valid.astype('float32'), Y_valid, verbose=0)
        print("Accuracy is: %.2f%%" % (scores[1]*100))
        accuracies += (scores[1]*100)


        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    final_accuracy = accuracies / nfolds
    print("Accuracy train independent avg: ", final_accuracy)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models


def run_cross_validation_process_test(info_string, models, X_test):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data = X_test
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)

    print test_res.shape

    # import code; code.interact(local=dict(globals(), **locals()))




num_folds = 3
info_string, models = run_cross_validation_create_models(num_folds, X, X_test, y)
run_cross_validation_process_test(info_string, models, X_test)