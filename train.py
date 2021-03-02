from stock_prediction import create_model, load_data, np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from parameters import *
import test
import matplotlib.pyplot as plt

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")

answer = input('Done save csv file. Do you want continue the training ?')

if answer == 'y':

    print('CSV file save at', ticker_data_filename)
    future_price_list = []

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
        checkpointer = ModelCheckpoint(os.path.join("results", model_name + str(number) + ".h5"),
                                       save_weights_only=True, save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name + str(number)))

        history = model.fit(data["X_train"], data["y_train"],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_data=(data["X_test"], data["y_test"]),
                            callbacks=[checkpointer, tensorboard],
                            verbose=1)
        model_path = os.path.join("results", model_name) + str(number) + ".h5"

        model.save(model_path)

        # load the data
        data = load_data(ticker, N_STEPS, lookup_step=number, test_size=TEST_SIZE,
                         feature_columns=FEATURE_COLUMNS, shuffle=False)

        # construct the model
        model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                             dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        # test

        print('Model', model_path)
        model.load_weights(model_path)

        # evaluate the model
        mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        # calculate the mean absolute error (inverse scaling)
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
        print("Mean Absolute Error:", mean_absolute_error)
        # predict the future price
        future_price = test.predict(model, data)
        future_price_list.append(future_price)
        print(f"Future price after {number} days is {future_price:.2f}$")
        accuracy_value = test.get_accuracy(model, data)
        print("Accuracy Score:", accuracy_value)

        # plot graph

        y_test = data["y_test"]
        X_test = data["X_test"]
        y_pred = model.predict(X_test)
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))

        plt.plot(y_test[-200:], c='b')
        plt.plot(y_pred[-200:], c='r')
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend(["Actual Price", "Predicted Price"])

        output_image = f"{ticker}_{date_now} For {number} day - Pred RM{future_price:.2f} - n_steps {N_STEPS} " \
                       f"- epoch {EPOCHS} - Acc {accuracy_value:.4f}.png"
        output_image_path = os.path.join("image_test", output_image)
        plt.savefig(output_image_path)
        print('output image path at', output_image_path)
        plt.close()


else:
    print('Training will not continue')
    print('CSV file save at', ticker_data_filename)

count = 0
for fp in future_price_list:
    count = + 1
    print(f'Day {count}: RM{fp:.2f}')
###########
