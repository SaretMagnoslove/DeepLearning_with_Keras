import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

# defining the constants
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'


# helper function for labeling the data
def classify(current, future):
    return 1 if float(future) > float(current) else 0


def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)
    sequencial_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])

        if len(prev_days) == SEQ_LEN:
            sequencial_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequencial_data)
    # balancing the data
    buys = []
    sells = []
    for seq, target in sequencial_data:
        sells.append([seq, target]) if target == 0 else buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys, sells = buys[:lower], sells[:lower]

    sequencial_data = buys + sells
    random.shuffle(sequencial_data)
    # splitting the data into X and y
    X = []
    y = []
    for seq, target in sequencial_data:
        X.append(seq)
        y.append(target)
    # returning X and y
    return np.array(X), y


# merging the close and volume of the datasets on timestump
main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    df = pd.read_csv(
        dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(
        columns={
            "close": f"{ratio}_close",
            "volume": f"{ratio}_volume"
        },
        inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    main_df = df if len(main_df) == 0 else main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(
    -FUTURE_PERIOD_PREDICT)
main_df['target'] = list(
    map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

# print(main_df[[f"{RATIO_TO_PREDICT}_close", 'future', 'target']].head(10))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05 * len(times))]
# print(last_5pct)
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)
# validating the data
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(
    f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}"
)
