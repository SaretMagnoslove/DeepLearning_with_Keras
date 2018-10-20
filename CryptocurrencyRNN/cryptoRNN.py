import pandas as pd 

# defining the constants
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'

# helper function for labeling the data
def classify(current, future):
    return 1 if float(future) > float(current) else 0  
    
# merging the close and volume of the datasets on timestump
main_df = pd.DataFrame()
ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]

for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset, names=["time",
                                     "low",
                                     "high",
                                     "open",
                                     "close",
                                     "volume"])
    df.rename(columns={"close": f"{ratio}_close",
                       "volume": f"{ratio}_volume"},
                        inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    main_df = df if len(main_df)==0 else main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

print(main_df[[f"{RATIO_TO_PREDICT}_close", 'future', 'target']].head(10))
    