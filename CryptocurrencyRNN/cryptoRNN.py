import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# defining the constants
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'BTC-USD'
EPOCHS = 10
BATCH_SIZE = 64
NAME = F"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


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
# building the model
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=train_x.shape[1:], return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=train_x.shape[1:]))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy', 
             optimizer=opt,
             metrics=['accuracy'])
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint]
)