import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import norm, chisquare
from adjustText import adjust_text

# Matplotlibフォント設定（日本語対応）
matplotlib.rc("font", family="SimHei")
plt.rcParams["axes.unicode_minus"] = False

# タイトル
st.title("プロジェクトモジュールデータ分析ツール")

# データ入力フォーマットの説明
st.write("以下のフォーマットでデータを入力してください：")
st.code(
    """
プロジェクト名 size = 総データ量
モジュール名1 size = サイズ1
モジュール名2 size = サイズ2
...
-------------------
別のプロジェクト名 size = 総データ量
モジュール名A size = サイズA
モジュール名B size = サイズB
...
    """
)

# ユーザー入力
user_input = st.text_area("ここにデータを入力してください:", height=400)

# 入力データ解析関数
def parse_input_data(input_text):
    projects = []
    current_project = None

    for line in input_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if "size =" in line and "-------------------" not in line:
            try:
                name, size = line.split("size =")
                name = name.strip()
                size = float(size.strip())

                if current_project is None:
                    current_project = {"プロジェクト名": name, "総データ量 (MB)": size, "モジュール": []}
                else:
                    current_project["モジュール"].append({"モジュール名": name, "サイズ (MB)": size})
            except ValueError:
                st.error("無効なデータ形式があります。フォーマットを確認してください。")
                return []

        elif "-------------------" in line:
            if current_project:
                projects.append(current_project)
                current_project = None

    if current_project:
        projects.append(current_project)

    return projects

# 小さいデータを「その他」にまとめる関数
def group_small_data(data, threshold=5):
    total_size = data["サイズ (MB)"].sum()
    grouped_data = data[data["サイズ (MB)"] >= (total_size * threshold / 100)]
    small_data = data[data["サイズ (MB)"] < (total_size * threshold / 100)]

    other_details = ""
    if not small_data.empty:
        other_details = ", ".join(small_data["プロジェクト名"])
        other_row = pd.DataFrame({
            "プロジェクト名": ["その他"],
            "サイズ (MB)": [small_data["サイズ (MB)"].sum()]
        })
        grouped_data = pd.concat([grouped_data, other_row], ignore_index=True)

    return grouped_data, other_details

# データ解析と描画
if user_input:
    projects = parse_input_data(user_input)

    if projects:
        st.write("## データ概要")
        all_modules = [{"プロジェクト名": p["プロジェクト名"], "モジュール名": m["モジュール名"], "サイズ (MB)": m["サイズ (MB)"]}
                       for p in projects for m in p["モジュール"]]
        df_modules = pd.DataFrame(all_modules)

        # 元のデータを表示
        st.write("### 元のデータ")
        st.dataframe(df_modules)

        # 統計情報
        st.write("### データの統計情報")
        st.write(f"平均: {df_modules['サイズ (MB)'].mean():.2f} MB")
        st.write(f"中央値: {df_modules['サイズ (MB)'].median():.2f} MB")
        st.write(f"最頻値: {df_modules['サイズ (MB)'].mode()[0]:.2f} MB")
        st.write(f"標準偏差: {df_modules['サイズ (MB)'].std():.2f} MB")
        st.write(f"最大値: {df_modules['サイズ (MB)'].max():.2f} MB")
        st.write(f"最小値: {df_modules['サイズ (MB)'].min():.2f} MB")

        # ヒストグラム + 正規分布 + 実際の分布
        st.write("### モジュールデータサイズの分布（ヒストグラム + 正規分布 + 実際の分布）")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df_modules["サイズ (MB)"], bins=20, kde=False, color="blue", stat="density", ax=ax, label="データ分布")
        mean = df_modules["サイズ (MB)"].mean()
        std_dev = df_modules["サイズ (MB)"].std()
        x = np.linspace(df_modules["サイズ (MB)"].min(), df_modules["サイズ (MB)"].max(), 100)
        normal_dist = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        ax.plot(x, normal_dist, color="red", label="正規分布")
        sns.kdeplot(df_modules["サイズ (MB)"], color="purple", linewidth=2, label="カーネル密度推定", ax=ax)
        y = np.full(len(df_modules["サイズ (MB)"]), -0.02)
        ax.scatter(df_modules["サイズ (MB)"], y, color="green", alpha=0.6, label="データ点", s=15)

        plt.xlabel("モジュールサイズ (MB)")
        plt.ylabel("密度")
        plt.title("モジュールサイズの分布と正規分布")
        plt.legend()
        st.pyplot(fig)

        # 箱ひげ図
        st.write("### プロジェクトごとのサイズ分布（箱ひげ図）")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_modules, x="プロジェクト名", y="サイズ (MB)", ax=ax, palette="Set3")
        plt.xticks(rotation=45)
        plt.title("プロジェクトごとのサイズ分布")
        st.pyplot(fig)

        # 累積分布関数（CDF）
        st.write("### 累積分布関数（CDF）")
        sorted_sizes = np.sort(df_modules["サイズ (MB)"])
        cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_sizes, cdf, marker=".", linestyle="none", color="orange")
        plt.xlabel("モジュールサイズ (MB)")
        plt.ylabel("累積割合")
        plt.title("累積分布関数（CDF）")
        st.pyplot(fig)

        # トップモジュールランキング
        st.write("### トップモジュールランキング")
        top_modules = df_modules.nlargest(5, "サイズ (MB)")
        st.write("上位 5 モジュール:")
        st.table(top_modules)

        # プロジェクトごとのモジュールサイズヒートマップ
        st.write("### プロジェクトごとのモジュールサイズヒートマップ")
        try:
            pivot_table = df_modules.pivot_table(
                index="モジュール名",
                columns="プロジェクト名",
                values="サイズ (MB)",
                fill_value=0
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            plt.title("モジュールサイズヒートマップ")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"ヒートマップ生成時にエラーが発生しました: {e}")

        # 総括円グラフ
        st.write("### データ比率 (プロジェクトごと)")
        try:
            project_sizes = pd.DataFrame({
                "プロジェクト名": [p["プロジェクト名"] for p in projects],
                "サイズ (MB)": [p["総データ量 (MB)"] for p in projects]
            })

            project_sizes, other_details = group_small_data(project_sizes, threshold=5)

            fig, ax = plt.subplots(figsize=(10, 10))
            wedges, texts, autotexts = ax.pie(
                project_sizes["サイズ (MB)"],
                labels=project_sizes["プロジェクト名"],
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("Set2", len(project_sizes)),
                wedgeprops={"edgecolor": "black", "linewidth": 0.5},
            )

            plt.title("プロジェクトごとのデータ比率", fontsize=16)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"総括円グラフ生成時にエラーが発生しました: {e}")

else:
    st.info("データを入力してください。")
