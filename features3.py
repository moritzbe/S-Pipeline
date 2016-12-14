
# coding: utf-8
import pandas as pd
import code



sample_path = '1percent.csv'

real_path = '../Receivables_all.csv'
receivable_headers = ["ALS_ID","ALG_ID","EXPECTED_LOSS","EXPECTED_LOSS_CUR_ID",
"UNEXPECTED_LOSS","UNEXPECTED_LOSS_CUR_ID","NOMINAL_AMOUNT","NOMINAL_AMOUNT_CUR_ID",
"REMAINING_AMOUNT","REMAINING_AMOUNT_CUR_ID","RECEIPT_DATE","DUE_DATE",
"TRADE_DISPUTE","ADD_ON_DAYS","INCURRED_LOSS",
"INCURRED_LOSS_CUR_ID","STATUS","STATUS_CURRENT",
"INSOLVENCY_DEFAULT_DATE"]
nt('Done: Reading the data')

df = pd.read_csv(real_path, header= None, sep=';', infer_datetime_format=True, usecols=[3, 4, 12, 13, 14, 15, 16, 17, 20, 21, 28, 29, 33, 37, 43, 44, 46, 53, 59], low_memory=True)
df.columns = receivable_headers



print('Done: Reading the data')

pd.concat([df,pd.DataFrame(columns=['EURO_AMOUNT_NOMINAL', 'EURO_AMOUNT_REMAINING'])])
# pd.concat([df,pd.DataFrame(columns=['EURO_AMOUNT_REMAINING'])])
currency_df = pd.read_csv('../FXRates.csv', sep=';', header=None, dayfirst=True)#[:].iloc[:, [0, 3]]
currency_df[3] = 1/currency_df[3]
dict_curr = currency_df.set_index(0)[3].to_dict()

df['EURO_AMOUNT_NOMINAL'] = df.apply(lambda x: x['NOMINAL_AMOUNT'] * dict_curr[x['NOMINAL_AMOUNT_CUR_ID']] if dict_curr.get(x['NOMINAL_AMOUNT_CUR_ID']) else x['NOMINAL_AMOUNT'], axis =1)
df['EURO_AMOUNT_REMAINING'] = df.apply(lambda x: x['REMAINING_AMOUNT'] * dict_curr[x['REMAINING_AMOUNT_CUR_ID']] if dict_curr.get(x['REMAINING_AMOUNT_CUR_ID']) else x['REMAINING_AMOUNT'], axis =1)





import numpy as np
print('Done: currency conv.')
df['RECEIPT_DATE'] = pd.to_datetime(df['RECEIPT_DATE']) 
df['DUE_DATE'] = pd.to_datetime(df['DUE_DATE']) 
grouped = df.groupby(by=['ALS_ID', df['RECEIPT_DATE'].map(lambda x: x.year)])

nominal = grouped['EURO_AMOUNT_NOMINAL'].sum().reset_index()
nominal = nominal.pivot(columns= 'RECEIPT_DATE', index='ALS_ID', values='EURO_AMOUNT_NOMINAL').fillna(0)
for x in nominal.columns:
    if  (x < 2013 or x > 2015 ):
        nominal.drop(x, axis=1, inplace=True)

remaining = grouped['EURO_AMOUNT_REMAINING'].sum().reset_index()
remaining = remaining.pivot(columns= 'RECEIPT_DATE', index='ALS_ID', values='EURO_AMOUNT_REMAINING').fillna(0)

for x in remaining.columns:
    print(type(x), x, x < 2013)
    if (x < 2013 or x > 2015 ):
        remaining.drop(x, axis=1, inplace=True)



# ### Ratings
print('Done: Nominal & Remaining, Doing Rating')



# ### TradeDisplay 
print('Done: Ratings')

pd.concat([df,pd.DataFrame(columns=['TRADEDISP'])])
df['TRADEDISP'] = df.apply(lambda x: 1 if ((x['TRADE_DISPUTE']==1) or  (x['TRADE_DISPUTE']==2)) else 0, axis =1)

tradedisp = df.groupby(by=['ALS_ID'])['TRADEDISP'].mean()


### Current Status ------ 3 Features
pd.concat([df,pd.DataFrame(columns=['BAD_STATUS_CURRENT'])])
df['BAD_STATUS_CURRENT'] = df.apply(lambda x: 1 if ((x['STATUS_CURRENT']==5) or (x['STATUS_CURRENT']==6) or (x['STATUS_CURRENT']==8)) else 0, axis =1)
st1 = df.groupby(by=['ALS_ID'])['BAD_STATUS_CURRENT'].mean()


pd.concat([df,pd.DataFrame(columns=['GOOD_STATUS_CURRENT'])])
df['GOOD_STATUS_CURRENT'] = df.apply(lambda x: 1 if ((x['STATUS_CURRENT']==3)) else 0, axis =1)
st2 = df.groupby(by=['ALS_ID'])['GOOD_STATUS_CURRENT'].mean()

pd.concat([df,pd.DataFrame(columns=['UNKNOWN_STATUS_CURRENT'])])
df['UNKNOWN_STATUS_CURRENT'] = df.apply(lambda x: 1 if ((x['STATUS_CURRENT']==2) or (x['STATUS_CURRENT']==4)) else 0, axis =1)
st3 = df.groupby(by=['ALS_ID'])['UNKNOWN_STATUS_CURRENT'].mean()

### Status ------ 3 Features
pd.concat([df,pd.DataFrame(columns=['BAD_STATUS'])])
df['BAD_STATUS'] = df.apply(lambda x: 1 if ((x['STATUS']==5) or (x['STATUS']==6) or (x['STATUS']==8)) else 0, axis =1)
st4 = df.groupby(by=['ALS_ID'])['BAD_STATUS'].mean()

pd.concat([df,pd.DataFrame(columns=['GOOD_STATUS'])])
df['GOOD_STATUS'] = df.apply(lambda x: 1 if ((x['STATUS']==3)) else 0, axis =1)
st5 = df.groupby(by=['ALS_ID'])['GOOD_STATUS'].mean()


pd.concat([df,pd.DataFrame(columns=['UNKNOWN_STATUS'])])
df['UNKNOWN_STATUS'] = df.apply(lambda x: 1 if ((x['STATUS']==2) or (x['STATUS']==4)) else 0, axis =1)
st6 = df.groupby(by=['ALS_ID'])['UNKNOWN_STATUS'].mean()


print('Done: All other features')


# ### Concat Features

features = pd.concat([nominal, remaining, tradedisp, st1, st2, st3, st4, st5, st6], axis=1).reset_index()
# features
print('Done: concat')

code.interact(local=dict(globals(), **locals()))

np.save("full_data_no_rating", features, allow_pickle=True, fix_imports=True)
np.savetxt("full4avi_no_rating.csv", features, delimiter=";")



