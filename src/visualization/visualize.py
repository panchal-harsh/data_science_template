import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl

df=pd.read_pickle("../../data/interim/01_data_preprocessed.pkl")


mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"]=(20,5)
mpl.rcParams["figure.dpi"]=100

'''

for label in df["label"].unique():
    subset=df[df["label"]==label]
    fig,ax=plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True),label=label)
    plt.show()
    

for label in df["label"].unique():
    subset=df[df["label"]==label]
    fig,ax=plt.subplots()
    plt.plot(subset[:100]["acc_y"],label=label)
    plt.show()
    

category_df=df.query("label=='squat'").query("participant=='A'").reset_index()


fig,ax=plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.setylabel="acc_y"
ax.set_xlabel="samples"
plt.legend()

participant_df=df.query("label=='bench'").sort_values("participant").reset_index()

fig,ax=plt.subplots()
participant_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel="acc_y"
ax.set_xlabel="samples"
plt.legend()

'''

labels=df["label"].unique()
participants=df["participant"].unique()

for label in labels:
   for participant in participants:
        all_axis_df=df.query(f"label=='{label}'").query(
            f"participant=='{participant}'"
            ).reset_index()
        fig,ax=plt.subplots()
        all_axis_df[['acc_y','acc_x','acc_z']].plot(ax=ax)
        ax.set_ylabel('acc_y')
        ax.set_xlabel("samples")
        plt.title(f"{label}({participant})".title())
        plt.legend() 
        

for label in labels:
   for participant in participants:
        all_axis_df=df.query(f"label=='{label}'").query(
            f"participant=='{participant}'"
            ).reset_index()
        fig,ax=plt.subplots()
        all_axis_df[['gyr_y','gyr_x','gyr_z']].plot(ax=ax)
        ax.set_ylabel('gyr_y')
        ax.set_xlabel("samples")
        plt.title(f"{label}({participant})".title())
        plt.legend() 




for label in labels:
   for participant in participants:
        combined_plot_df=df.query(f"label=='{label}'").query(
            f"participant=='{participant}'"
            ).reset_index()
        fig,ax=plt.subplots(nrows=2,sharex=True,figsize=(20,10))
        combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
        combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

        ax[0].legend(
            loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True
        )

        ax[1].legend(
            loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,shadow=True
        )
        ax[1].set_xlabel("samples")
        
        plt.savefig(f"../../reports/figures/{label}({participant}).png")
        plt.show()





