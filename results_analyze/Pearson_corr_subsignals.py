import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
root_path = os.path.dirname(os.path.abspath('__file__'))
# root_path = os.path.abspath(os.path.join(root_path,os.path.pardir))
graph_path = root_path+'/graph/'
print("root path:{}".format(root_path))
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [7.48, 5.61]
# plt.rcParams['image.cmap']='plasma'
plt.rcParams['image.cmap']='viridis'
# plt.rcParams['axes.linewidth']=0.8


yx_vmd_train = pd.read_csv(root_path+"/yx_vmd/data/VMD_TRAIN.csv")
yx_eemd_train = pd.read_csv(root_path+"/yx_eemd/data/EEMD_TRAIN.csv")
yx_dwt_train = pd.read_csv(root_path+'/yx_wd/data/db45-3/WD_TRAIN.csv')

zjs_vmd_train = pd.read_csv(root_path+"/zjs_vmd/data/VMD_TRAIN.csv")
zjs_eemd_train = pd.read_csv(root_path+"/zjs_eemd/data/EEMD_TRAIN.csv")
zjs_dwt_train = pd.read_csv(root_path+'/zjs_wd/data/db45-3/WD_TRAIN.csv')

yx_vmd_train=yx_vmd_train.drop("ORIG",axis=1)
yx_eemd_train=yx_eemd_train.drop("ORIG",axis=1)
yx_dwt_train=yx_dwt_train.drop("ORIG",axis=1)
zjs_vmd_train=zjs_vmd_train.drop("ORIG",axis=1)
zjs_eemd_train=zjs_eemd_train.drop("ORIG",axis=1)
zjs_dwt_train=zjs_dwt_train.drop("ORIG",axis=1)

yx_vmd_corrs = yx_vmd_train.corr(method="pearson")
yx_eemd_corrs = yx_eemd_train.corr(method="pearson")
yx_dwt_corrs = yx_dwt_train.corr(method="pearson")
zjs_vmd_corrs = zjs_vmd_train.corr(method="pearson")
zjs_eemd_corrs = zjs_eemd_train.corr(method="pearson")
zjs_dwt_corrs = zjs_dwt_train.corr(method="pearson")

# corrs=[
#     abs(zjs_vmd_corrs),abs(yx_vmd_corrs),
#     abs(zjs_eemd_corrs),abs(yx_eemd_corrs),
#     abs(zjs_dwt_corrs),abs(yx_dwt_corrs),
#     ]

# titles=[
#     "VMD'subsignals at ZJS","VMD'subsignals at YX",
#     "EEMD'subsignals at ZJS","EEMD'subsignals at YX",
#     "DWT(db45-3)'subsignals at ZJS","DWT(db45-3)'subsignals at YX",
# ]
corrs=[
    abs(zjs_vmd_corrs),abs(zjs_eemd_corrs),abs(zjs_dwt_corrs),
    abs(yx_vmd_corrs),abs(yx_eemd_corrs),abs(yx_dwt_corrs),
    ]
titles=[
    "VMD subsignals at ZJS","EEMD subsignals at ZJS","DWT(db45-3) subsignals at ZJS",
    "VMD subsignals at YX","EEMD subsignals at YX","DWT(db45-3) subsignals at YX",
]


fig=plt.figure(figsize=(7.48,5.))
for i in range(len(corrs)):
    plt.subplot(2,3,i+1)
    plt.title(titles[i])
    sign_num=corrs[i].shape[1]
    ticks = list(range(sign_num))
    labels=[]
    for j in ticks:
        if titles[i].find('VMD')>=0:
            labels.append(r'$IMF_{'+str(j+1)+'}$')
        elif titles[i].find('EEMD')>=0:
            if j==sign_num-1:
                labels.append(r'$R$')
            else:
                labels.append(r'$IMF_{'+str(j+1)+'}$')
        elif titles[i].find('DWT')>=0:
            if j==sign_num-1:
                labels.append(r'$A_{'+str(j)+'}$')
            else:
                labels.append(r'$D_{'+str(j+1)+'}$')
    print(sign_num)
    print(labels)
    im=plt.imshow(corrs[i])
    plt.xticks(ticks=ticks,labels=labels,rotation=90)
    plt.yticks(ticks=ticks,labels=labels)
    plt.xlim(-0.5,sign_num-0.5)
    plt.ylim(-0.5,sign_num-0.5)
    # plt.ylabel(r"${S}_j$")
    # plt.xlabel(r"${S}_i$")
    if i==2:
        cb_ax = fig.add_axes([0.91, 0.09, 0.03, 0.86])#[left,bottom,width,height]
        cbar = fig.colorbar(im, cax=cb_ax)
        im.colorbar.set_label(r"$Pearson\ correlation\ coefficient$")
        # plt.colorbar(ax.colorbar, fraction=0.045)
        plt.clim(0,1)
    # plt.colorbar(im.colorbar, fraction=0.045)
    # im.colorbar.set_label("$Corr_{i,j}$")
    # plt.clim(0,1)
fig.subplots_adjust(bottom=0.08, top=0.96, left=0.06, right=0.9,wspace=0.3, hspace=0.35)
# plt.savefig(graph_path+"nse_wd.eps",format="EPS",dpi=2000)
plt.savefig(graph_path+"Pearson_corr_subsignals_imshow.tif",format="TIFF",dpi=1200)
# plt.savefig(graph_path+"Pearson_corr_subsignals_imshow.eps",format="EPS",dpi=2000)
plt.show()
