import train
import test

anw = input('y = train new data. n = test using previous model:\n')
if anw == 'y':
    train.training()

else:

    list_model_path = []
    test.result(list_model_path)
