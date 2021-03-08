‘stock_filter.py’ is to find all green stock and hammer candle stick for now in the script i grab the data from top 50 high volume stock at [Bursa Malaysia]( https://www.bursamalaysia.com/market_information/shariah_compliant_equities_prices?top_stock=top_active&per_page=50&page=1) website

The stock prediction program is referred from the [tutorial] (https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras) and you can see the original code [here]( https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction). This program was edited for make train and test for few days in one run

First, you need to choose what stock you want to test using this script at yahoo finance and identify the code stock. Then change in ‘parameters.py ‘at ‘ticker’. At same file change at ‘LOOKUP_STEP‘  parameter how many days you want to predict

Run ‘main.py’

If you want to stop the program after train, model path will save at ‘results/latest_model.txt’. then you can continue test the model by running again the ‘main.py’ file

Each day train will be producing model and for the test will be produce predicted price

From my testing the getting more days you want predict, the more accuracy it gets. For example, if your ‘LOOKUP_STEP = 20’, days 20 more gain more accuracy from previous days.

‘EPOCHS = 800’ and ‘N_STEPS = 90’ is the best parameter for my testing. You also can play around with ‘N_STEPS’, ‘EPOCHS’ and ‘LOOKUP_STEP’ see which combination works best.
