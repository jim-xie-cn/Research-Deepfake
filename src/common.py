import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore, expon, kstest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from scipy.stats import zscore
from tqdm.auto import tqdm
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, normaltest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from itertools import combinations
from FreeAeonML.FADataEDA import CFADataDistribution,CFADataTest,CFAFitter,CFACommonStats
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=FutureWarning, message="DataFrameGroupBy.apply operated on the grouping columns")

def get_normal(df_data, group='kind'):
    columns_to_standardize = df_data.select_dtypes(include=[np.number]).columns.tolist()
    df_standardized = df_data.copy()

    def safe_zscore(x):
        valid_mask = ~np.isnan(x)
        if valid_mask.sum() <= 1:
            return np.zeros_like(x)
        std = np.nanstd(x)
        if std == 0:
            return np.zeros_like(x)
        z = (x - np.nanmean(x)) / std
        return z
    df_standardized[columns_to_standardize] = (
        df_data.groupby(group)[columns_to_standardize].transform(safe_zscore)
    )
    return df_standardized
    
def get_stats(df_data):
    stats = CFACommonStats.get_stats(df_data)
    all_data = []
    for key in stats:
        item = stats[key]
        item['item'] = key
        all_data.append(item)
    return pd.DataFrame(all_data)
    
def show_stats(df_data,group=None,bins=100, title = "",cols = ["cv", "kurt_pearson", "kurt_fisher", "skew", "mean", "std"]):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        if group != None:
            sns.histplot(data=df_data,x=col,hue=group,bins=bins,stat="probability",common_norm=False,kde=False,ax=axes[i])
        else:
            sns.histplot(data=df_data,x=col,bins=bins,stat="probability",common_norm=False,kde=True,ax=axes[i])
        axes[i].set_title(title + col)
    plt.tight_layout()
    
def show_t_sne(df,group="kind"):
    features = pd.DataFrame()
    for key in df.keys():
        if key != group:
            features[key] = df[key]   
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_scaled)
    df['tsne1'] = tsne_results[:, 0]
    df['tsne2'] = tsne_results[:, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='kind', palette='coolwarm', s=5)
    
    plt.title('t-SNE of Data with kind as hue')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='kind')
    plt.tight_layout()

def show_pca(df,group = 'kind'):
    features = pd.DataFrame()
    for key in df.keys():
        if key != group:
            features[key] = df[key]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features_scaled)
    
    df['pca1'] = pca_results[:, 0]
    df['pca2'] = pca_results[:, 1]
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='kind', palette='coolwarm', s=100)
    
    plt.title('PCA of Data with kind as hue')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title=group)
    plt.tight_layout()

def show_mfs(df_tmp,group='kind'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    plots = [
        ('q', 't(q)', "q vs t(q)"),
        ('q', 'a(q)', "q vs a(q)"),
        ('q', 'd(q)', "q vs d(q)"),
        ('q', 'f(a)', "q vs f(a)"),
        ('a(q)', 'f(a)', "a(q) vs f(a)"),
        ('a(q)', 'd(q)', "a(q) vs d(q)")
    ]

    for i, (x, y, title) in enumerate(plots):
        sns.lineplot(
            data=df_tmp, x=x, y=y, hue=group,
            palette='coolwarm',  
            ax=axes[i]
        )
        axes[i].set_title(title)

    plt.tight_layout()
    
def show_hist(df_tmp, group="kind",bins=10, cols = ['t(q)', 'a(q)', 'd(q)', 'f(a)']):
    n = len(cols)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(
            data=df_tmp,
            x=col,
            hue=group,
            bins=bins,
            stat="probability",
            common_norm=False,
            palette='coolwarm',
            element="step",      
            ax=axes[i]
        )
        axes[i].set_title(f"P({col})")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

def test_data(df_data, y_col="kind", x_cols=None):
    if not x_cols:
        x_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()

    y_vals = df_data[y_col].unique().tolist()
    results = []

    for col in x_cols:
        if len(y_vals) > 2:
            groups = [df_data[df_data[y_col] == val][col].dropna() for val in y_vals]
            normal_pvals = [normaltest(g).pvalue for g in groups if len(g) > 2]
            if all(p > 0.05 for p in normal_pvals):
                stat, p = f_oneway(*groups)
                method = "ANOVA"
            else:
                stat, p = kruskal(*groups)
                method = "Kruskal-Wallis"
            results.append({"feature": col, "method": method, "p_value": p})

        else:
            for i, j in combinations(y_vals, 2):
                g1 = df_data[df_data[y_col] == i][col].dropna()
                g2 = df_data[df_data[y_col] == j][col].dropna()
                p1 = normaltest(g1).pvalue if len(g1) > 2 else 1
                p2 = normaltest(g2).pvalue if len(g2) > 2 else 1
                if p1 > 0.05 and p2 > 0.05:
                    stat, p = ttest_ind(g1, g2, nan_policy='omit')
                    method = "t-test"
                else:
                    stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                    method = "Mann-Whitney U"
                results.append({"feature": col, "group1": i, "group2": j, "method": method, "p_value": p})

    print("\n===Testing ===")
    X = df_data[x_cols]
    y = LabelEncoder().fit_transform(df_data[y_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    
    acc = clf.score(X_test, y_test)
    if acc > 0.7:
        print(f"\n acc: {acc:.3f} → High")
    elif acc > 0.55:
        print(f"\n acc: {acc:.3f} → Middle")
    else:
        print(f"\n:acc: {acc:.3f} → Low")
    
    return pd.DataFrame(results)

