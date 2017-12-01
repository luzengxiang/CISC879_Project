FOR /L %%a IN (4,1,7) DO (
python data_preprocess.py %%a
python train.py %%a 200
)

