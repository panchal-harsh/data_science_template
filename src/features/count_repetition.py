import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error


pd.options.mode.chained_assignment=None
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2              


df=pd.read_pickle("../../data/interim/01_data_preprocessed.pkl")
df=df[df["label"]!="rest"]

df_squared=df.copy()
acc_r=df_squared["acc_x"]**2+df_squared["acc_y"]**2+df_squared["acc_z"]**2
gyr_r=df_squared["gyr_x"]**2+df_squared["gyr_y"]**2+df_squared["gyr_z"]**2

df_squared["acc_r"]=np.sqrt(acc_r)
df_squared["gyr_r"]=np.sqrt(gyr_r)
df=df_squared.copy()

bench_df=df[df["label"]=="bench"]
squat_df=df[df["label"]=="squat"]
row_df=df[df["label"]=="bench"]
ohp_df=df[df["label"]=="ohp"]
dead_df=df[df["label"]=="dead"]


fs=1000/200
LowPass=LowPassFilter()

benchset=bench_df[bench_df["set"]==bench_df["set"].unique()[0]]
squatset=squat_df[squat_df["set"]==squat_df["set"].unique()[0]]
rowset=row_df[row_df["set"]==row_df["set"].unique()[0]]
ohpset=ohp_df[ohp_df["set"]==ohp_df["set"].unique()[0]]
deadset=dead_df[dead_df["set"]==dead_df["set"].unique()[0]]

benchset["acc_r"].plot()

def count_reps(dataset,cutoff=0.4,order=10,column="acc_r"):
    data=LowPass.low_pass_filter(
        dataset,col=column,sampling_frequency=fs,cutoff_frequency=cutoff,order=order
    )
    indexes=argrelextrema(data[column+"_lowpass"].values,np.greater)
    peaks=data.iloc[indexes]
    
    fig,ax=plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"],"o",color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise=dataset["label"].iloc[0].title()
    category=dataset["category"].iloc[0].title()
    plt.title(f"{category}{exercise}:{len(peaks)} reps")
    return len(peaks)


count_reps(benchset,cutoff=0.4)
count_reps(squatset,cutoff=0.35)
count_reps(rowset,cutoff=0.65,column="gyr_x")
count_reps(ohpset,cutoff=0.35)
count_reps(deadset,cutoff=0.4)

df["reps"]=df["category"].apply(lambda x:5 if x=="heavy" else 10)
rep_df=df.groupby(["label","category","set"])["reps"].max().reset_index()


for s in df["set"].unique():
    subset=df[df["set"]==s]
    column="acc_r"
    cutoff=0.4
    if subset["label"].iloc[0]=="squat":
        cutoff=0.35
    if subset["label"].iloc[0]=="row":
       cutoff=0.65
       col="gyr_x"
    if subset["label"].iloc[0]=="ohp":
        cutoff=0.35
    reps=count_reps(subset,cutoff=cutoff,column=column)
    rep_df.loc[rep_df["set"]==s,"reps_pred"]=reps
    
del rep_df["pres_pred"]                             
rep_df.groupby(["label","category"])[["reps","reps_pred"]].mean().plot.bar()
