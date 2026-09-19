from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["BiasOnly", "BiasMF", "CosineMF", "NormalizedMF"]


def save_all(fig, outbase):
    fig.savefig(str(outbase) + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(str(outbase) + '.pdf', bbox_inches='tight')
    fig.savefig(str(outbase) + '.tiff', dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--figures-dir', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    s100 = pd.read_csv(args.results_dir / 'ML100K_validated_summary.csv')
    s1m = pd.read_csv(args.results_dir / 'ML1M_validated_summary.csv')
    ablation = pd.read_csv(args.results_dir / 'five_dataset_direct_ablation.csv')
    retention = pd.read_csv(args.results_dir / 'retention_metadata.csv')
    sensitivity = pd.read_csv(args.results_dir / 'ML100K_cosine_norm_sensitivity_summary.csv')

    # Figure 1: primary MovieLens RMSE comparison
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.75), sharex=True)
    markers = {'BiasOnly':'o','BiasMF':'^','CosineMF':'s','NormalizedMF':'D'}
    lines = {'BiasOnly':':','BiasMF':'--','CosineMF':'-.','NormalizedMF':'-'}
    for ax, summary, title in zip(axes, [s100, s1m], ['MovieLens 100K','MovieLens 1M']):
        # handle either retain_frac or retention columns
        rcol = 'retain_frac' if 'retain_frac' in summary.columns else 'retention'
        for model in MODELS:
            ss = summary[summary.model == model].sort_values(rcol, ascending=False)
            x = ss[rcol].to_numpy() * 100
            y = ss.rmse_mean.to_numpy()
            sd = ss.rmse_sd.to_numpy()
            ax.plot(x, y, marker=markers[model], linestyle=lines[model], linewidth=1.6,
                    markersize=5, label=model)
            ax.fill_between(x, y-sd, y+sd, alpha=0.10)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel('Training ratings retained (nominal %)')
        ax.grid(axis='y', alpha=0.22)
        ax.invert_xaxis()
    axes[0].set_ylabel('Test RMSE (lower is better)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.02), ncol=4, frameon=False)
    fig.tight_layout(rect=[0,0,1,0.90], w_pad=1.2)
    save_all(fig, args.figures_dir / 'Figure1_cross_dataset_RMSE')

    # Figure 2: five-dataset direct-ablation heatmap
    order = ['MovieLens100K','MovieLens1M','BookCrossing','FilmTrust','Jester']
    labels = ['MovieLens 100K','MovieLens 1M','Book-Crossing','FilmTrust','Jester']
    ret = [1.00,0.75,0.50,0.25,0.10]
    mat = np.full((len(order),len(ret)),np.nan)
    for i,d in enumerate(order):
        for j,r in enumerate(ret):
            x=ablation[(ablation.dataset==d)&(np.isclose(ablation.nominal_retention,r))]
            if not x.empty:
                mat[i,j]=x.iloc[0].relative_rmse_gain_pct

    pos_cmap = LinearSegmentedColormap.from_list(
        'paper_teal', ['#F6F8F9','#D9ECEB','#A7D5D1','#66B5AE','#2E8F89','#176B70','#0B4F5C']
    )
    neg_fill, neg_edge = '#F3D9D6', '#A84E46'
    text_dark, separator = '#172126', '#69757A'
    fig, ax = plt.subplots(figsize=(10.8,4.7))
    norm = PowerNorm(gamma=0.55,vmin=0,vmax=max(1.8,float(np.nanmax(mat))))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v=mat[i,j]
            if v>=0:
                fc=pos_cmap(norm(v)); ec='white'
            else:
                fc=neg_fill; ec=neg_edge
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=fc,edgecolor=ec,linewidth=1.6))
            if v<0:
                color=neg_edge; weight='semibold'
            else:
                color='white' if norm(v)>0.62 else text_dark
                weight='semibold' if abs(v)>=0.45 else 'normal'
            ax.text(j,i,f'{v:+.2f}%',ha='center',va='center',fontsize=10.8,color=color,fontweight=weight)
    for i,j in np.argwhere(mat<0):
        ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor=neg_edge,linewidth=2.1))
    ax.set_xlim(-.5,4.5); ax.set_ylim(4.5,-.5)
    ax.set_xticks(range(5),['100','75','50','25','10'],fontsize=10.5)
    ax.set_yticks(range(5),labels,fontsize=10.7)
    ax.set_xlabel('Training ratings retained (nominal %)',fontsize=11,labelpad=8)
    ax.set_ylabel('Dataset',fontsize=11,labelpad=10)
    ax.axhline(1.5,color=separator,linewidth=1.35)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    sm=plt.cm.ScalarMappable(norm=norm,cmap=pos_cmap); sm.set_array([])
    cbar=fig.colorbar(sm,ax=ax,fraction=0.035,pad=0.025)
    cbar.set_label('Relative RMSE gain (%)',fontsize=10.5,labelpad=8)
    cbar.set_ticks([0,.1,.25,.5,1.0,1.5,1.8])
    cbar.ax.tick_params(labelsize=9); cbar.outline.set_linewidth(.6)
    fig.subplots_adjust(left=.19,right=.91,bottom=.18,top=.96)
    save_all(fig, args.figures_dir / 'Figure2_direct_ablation_heatmap')

    # Figure 3: deviation between realized and nominal retention
    fig, ax = plt.subplots(figsize=(9.1,4.7))
    styles = {
        'MovieLens 100K':dict(marker='o',linestyle='-'),
        'MovieLens 1M':dict(marker='s',linestyle='--'),
        'Book-Crossing':dict(marker='^',linestyle='-'),
        'FilmTrust':dict(marker='D',linestyle='-.'),
        'Jester':dict(marker='P',linestyle=':'),
    }
    for name in styles:
        t=retention[retention.dataset==name].sort_values('nominal_retention')
        dev=(t.actual_retention-t.nominal_retention)*100
        ax.plot(t.nominal_retention*100,dev,label=name,linewidth=2.1,markersize=7,**styles[name])
    for name in ['Book-Crossing','FilmTrust']:
        t=retention[(retention.dataset==name)&(np.isclose(retention.nominal_retention,0.10))].iloc[0]
        dev=(t.actual_retention-t.nominal_retention)*100
        ax.annotate(f'{t.actual_retention*100:.1f}% actual',xy=(10,dev),xytext=(5,4),
                    textcoords='offset points',fontsize=9)
    ax.axhline(0,linewidth=1.0,color='black',alpha=0.65)
    ax.set_xticks([10,25,50,75,100])
    ax.set_xlabel('Nominal training retention (%)')
    ax.set_ylabel('Realized minus nominal retention\n(percentage points)')
    ax.legend(ncol=3,loc='upper right',frameon=False,fontsize=9)
    ax.grid(axis='y',alpha=0.22)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.subplots_adjust(left=0.13,right=0.98,bottom=0.16,top=0.95)
    save_all(fig,args.figures_dir/'Figure3_realized_retention_deviation')

    # Figure 4: MovieLens 100K sensitivity heatmap (single 3 x 9 matrix)
    wds=[1e-5,1e-4,1e-3]; ks=[16,32,64]; retentions=[1.0,0.5,0.1]
    arr=[]
    for r in retentions:
        row=[]
        for wd in wds:
            for k in ks:
                x=sensitivity[np.isclose(sensitivity.retain_frac,r)&np.isclose(sensitivity.K,k)&np.isclose(sensitivity.weight_decay,wd)]
                row.append(float(x.gain_mean.iloc[0]))
        arr.append(row)
    arr=np.asarray(arr)
    pos_cmap = LinearSegmentedColormap.from_list(
        'paper_teal', ['#F6F8F9','#D9ECEB','#A7D5D1','#66B5AE','#2E8F89','#176B70','#0B4F5C']
    )
    text_dark, separator = '#172126', '#69757A'
    fig,ax=plt.subplots(figsize=(11.2,3.8))
    norm4=PowerNorm(gamma=0.8,vmin=0,vmax=.013)
    im=ax.imshow(arr,cmap=pos_cmap,norm=norm4,aspect='auto',interpolation='nearest')
    ax.set_xticks(np.arange(-.5,9,1),minor=True)
    ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5)
    ax.tick_params(which='minor',bottom=False,left=False)
    for x in [2.5,5.5]: ax.axvline(x,color=separator,lw=1.7)
    for i in range(3):
        for j in range(9):
            v=arr[i,j]
            color='white' if norm4(v)>0.60 else text_dark
            ax.text(j,i,f'{v:+.4f}',ha='center',va='center',fontsize=9.8,color=color,
                    fontweight='semibold' if v>=.009 else 'normal')
    ax.set_yticks(range(3),['100%','50%','10%'],fontsize=10.5)
    ax.set_ylabel('Training retention',fontsize=11,labelpad=9)
    ax.set_xticks(range(9),[str(k) for wd in wds for k in ks],fontsize=10)
    ax.set_xlabel('Latent dimension K (within each weight-decay group)',fontsize=10.8,labelpad=9)
    for center,label in zip([1,4,7],[r'Weight decay = $10^{-5}$',r'Weight decay = $10^{-4}$',r'Weight decay = $10^{-3}$']):
        ax.text(center,-.78,label,ha='center',va='bottom',fontsize=10.4,fontweight='semibold',color=text_dark,clip_on=False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    cbar=fig.colorbar(im,ax=ax,fraction=.028,pad=.025)
    cbar.set_label('RMSE gain\n(CosineMF - NormalizedMF)',fontsize=10,labelpad=8)
    cbar.ax.tick_params(labelsize=9); cbar.outline.set_linewidth(.6)
    fig.subplots_adjust(left=.12,right=.89,bottom=.22,top=.78)
    save_all(fig,args.figures_dir/'Figure4_sensitivity_heatmaps')


if __name__=='__main__':
    main()
