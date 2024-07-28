import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstractionX import FourierTransformationx
from sklearn.cluster import KMeans

df=pd.read_pickle("../../data/interim/02_outliers_chauvents.pkl")
predictor_columns=list(df.columns[:6])


for col in predictor_columns:
      df[col]=df[col].interpolate()

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"]=(20,5)
plt.rcParams["figure.dpi"]=100
plt.rcParams["lines.linewidth"]=2


for s in df['set'].unique():
  start= df[df['set']==s].index[0]
  stop=df[df["set"]==s].index[-1] 
  duration=stop-start
  df.loc[(df["set"]==s),'duration']=duration.seconds
  


duration_df=df.groupby(["category"])["duration"].mean()    

duration_df.iloc[0]/5
duration_df.iloc[1]/10

df_lowpass=df.copy()
LowPass=LowPassFilter()

fs=1000/200
cutoff=1.4

df_lowpass=LowPass.low_pass_filter(df_lowpass,"acc_z",fs,cutoff,order=5)
subset=df_lowpass[df_lowpass["set"]==45]


fig,ax=plt.subplots(nrows=2,sharex=True,figsize=(20,10))
ax[0].plot(subset["acc_z"].reset_index(drop=True),label="raw data")
ax[1].plot(subset["acc_z_lowpass"].reset_index(drop=True),label="filtered data")



for col in predictor_columns:
  df_lowpass=LowPass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5)
  df_lowpass[col]=df_lowpass[col+'_lowpass']
  del df_lowpass[col+'_lowpass']
  

df_pca=df_lowpass.copy()
PCA=PrincipalComponentAnalysis()
pc_values=PCA.determine_pc_explained_variance(df_pca,predictor_columns)

plt.figure(figsize=(10,10))
plt.plot(range(1,len(predictor_columns)+1),pc_values)
plt.xlabel("pricipal component number")
plt.ylabel("explained variance")


df_pca=PCA.apply_pca(df_pca,predictor_columns,3)

subset=df_pca[df_pca["set"]==35]
subset[["pca_1","pca_2","pca_3"]].plot()

df_squared=df_pca.copy()
acc_r=df_squared["acc_x"]**2+df_squared["acc_y"]**2+df_squared["acc_z"]**2
gyr_r=df_squared["gyr_x"]**2+df_squared["gyr_y"]**2+df_squared["gyr_z"]**2

df_squared["acc_r"]=np.sqrt(acc_r)
df_squared["gyr_r"]=np.sqrt(gyr_r)
subset=df_squared[df_squared["set"]==14]
subset[["acc_r","gyr_r"]].plot(subplots=True)


df_temporal=df_squared.copy()
NumAbs=NumericalAbstraction()

predictor_columns=predictor_columns+["acc_r","gyr_r"]

ws=int(1000/200)

for col in predictor_columns:
    
    df_temporal=NumAbs.abstract_numerical(df_temporal,[col],ws,"mean")
    df_temporal=NumAbs.abstract_numerical(df_temporal,[col],ws,"std")



df_temporal_list=[]
for s in df_temporal["set"].unique():
    subset=df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_columns:
        subset=NumAbs.abstract_numerical(subset,[col],ws,"mean")
        subset=NumAbs.abstract_numerical(subset,[col],ws,"std")
    df_temporal_list.append(subset)
    
df_temporal=pd.concat(df_temporal_list)
df_temporal.info()

#for g in df_temporal.groupby('set'):
#     g[1][["acc_y","acc_y_temp_mean_ws_5","acc_y_temp_std_ws_5"]].plot()
    
df_freq=df_temporal.copy().reset_index(drop=False)
FreqAbs=FourierTransformationx()

fs=int(1000/200)
ws=int(2800/200)




df_freq_list=[]
for s in df_freq["set"].unique():
  print(f"Applying fourier transformationto set {s}")
  subset=df_freq[df_freq["set"]==s].reset_index(drop=True).copy()
  subset=FreqAbs.abstract_frequency(subset,predictor_columns,ws,fs)
  df_freq_list.append(subset)

df_freq=pd.concat(df_freq_list).set_index("epoch (ms)",drop=True)

df_freq=df_freq.dropna()
df_freq=df_freq.iloc[::2]



df_cluster=df_freq.copy()
cluster_columns=["acc_x","acc_y","acc_z"]
k_values=range(2,10)
inertias= []

for k in k_values:
  subset=df_cluster[cluster_columns] 
  kmeans=KMeans(n_clusters=k,n_init=20,random_state=0)
  cluster_labels=kmeans.fit_predict(subset)
  inertias.append(kmeans.inertia_)


plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("k")
plt.ylabel("inertial")

#number of cluster=5

subset=df_cluster[cluster_columns] 
kmeans=KMeans(n_clusters=5,n_init=20,random_state=0)
cluster_labels=kmeans.fit_predict(subset)
inertias.append(kmeans.inertia_)
df_cluster["cluster"]=cluster_labels


fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
  subset=df_cluster[df_cluster["cluster"]==c]
  ax.scatter(subset["acc_x"],subset["acc_y"],subset["acc_z"],label=c)
ax.set_xlabel=("x")
ax.set_ylabel=("y")
ax.set_zlabel=("z")
plt.legend()

fig=plt.figure(figsize=(15,15))
ax=fig.add_subplot(projection="3d")
for c in df_cluster["label"].unique():
  subset=df_cluster[df_cluster["label"]==c]
  ax.scatter(subset["acc_x"],subset["acc_y"],subset["acc_z"],label=c)
ax.set_xlabel=("x")
ax.set_ylabel=("y")
ax.set_zlabel=("z")
plt.legend()


df_cluster.to_pickle("../../data/interim/03_data_features.pkl")