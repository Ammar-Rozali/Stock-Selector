from stock_prediction import load_data, create_model, np
from parameters import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(
        map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(
        map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)


def predict(model, data, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:N_STEPS]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price


def plot_graph(model, data, number, future_price, accuracy_value):
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


def result(list_model_path, answer=None):
    number = 0
    if answer:
        model_path = list_model_path
    else:
        model_path = []

        with open(os.path.join("results", 'latest_model.txt'), 'r') as f:
            print('collect model path in txt file')
            filecontents = f.readlines()

            for line in filecontents:
                # remove linebreak which is the last character of the string
                current_place = line[:-1]

                # add item to the list
                model_path.append(current_place)

    future_price_list = []
    for mp in model_path:
        # load the data
        number += 1
        data = load_data(ticker, N_STEPS, lookup_step=number, test_size=TEST_SIZE,
                         feature_columns=FEATURE_COLUMNS, shuffle=False)

        # construct the model
        model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                             dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        print('Model', mp)
        model.load_weights(mp)

        # evaluate the model
        mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
        # calculate the mean absolute error (inverse scaling)
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
        print("Mean Absolute Error:", mean_absolute_error)
        # predict the future price
        future_price = predict(model, data)
        future_price_list.append(future_price)

        print(f"Future price after {number} days is {future_price:.2f}$")
        accuracy_value = get_accuracy(model, data)
        print("Accuracy Score:", accuracy_value)

        plot_graph(model, data, number, future_price, accuracy_value)
        count = 0

    for fp in future_price_list:
        count += 1
        print(f'Day {count}: RM{fp:.2f}')
