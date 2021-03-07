from stock_prediction import create_model, load_data, np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from parameters import *
import test

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")


def training():

    list_model_path = []
    for number in (len + 1 for len in range(LOOKUP_STEP)):
        print("number", number)
        # load the data
        data = load_data(ticker, N_STEPS, lookup_step=number, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
        # last_month=last_month)

        # save the dataframe/csv
        data["df"].to_csv(ticker_data_filename)

        # construct the model
        model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                             dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        # some tensorflow callbacks
        checkpointer = ModelCheckpoint(os.path.join("results", model_name + f'-Days-{number}-epoch{EPOCHS}.h5'),
                                       save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name + f'-Days-{number}-epoch{EPOCHS}'))

        history = model.fit(data["X_train"], data["y_train"],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(data["X_test"], data["y_test"]),
                            callbacks=[checkpointer, tensorboard],
                            verbose=1)
        model_path = os.path.join("results", model_name) + f'-Days-{number}-epoch{EPOCHS}.h5'

        model.save(model_path)

        list_model_path.append(model_path)
    print('CSV file save at', ticker_data_filename)


    answer = input('Do you want continue the test and result ?')

    if answer == 'y':

        test.result(list_model_path, answer)
    else:
        with open(os.path.join("results", 'latest_model.txt'), 'w') as f:
            for mp in list_model_path:
                f.write('%s\n' % mp)

        print('model path save at', os.path.join("results", 'latest_model.txt'))


###########
