import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.font_manager as fm

#current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)
# フォントの相対パスを指定（スクリプトと同じディレクトリに「font」フォルダを作成し、その中にフォントを配置）
#font_path = os.path.join(current_dir, "font", "SimHei.ttf")

font_path = "./data/SimHei.ttf"
font_prop = fm.FontProperties(fname=font_path)

# Matplotlibの設定に適用
# matplotlib.rc("font", family=font_prop.get_name())
# SimHei フォントをシステムインストール済みフォントで指定
# matplotlib.rc("font", family="SimHei")
# グローバルにフォントを設定
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False


# 标题
st.title("项目模块数据分析工具")


# 数据输入格式说明
st.write("请按照以下格式输入数据：")
st.code(
    """
项目名 size = 总数据量
模块名1 size = 大小1
模块名2 size = 大小2
...
-------------------
另一个项目名 size = 总数据量
模块名A size = 大小A
模块名B size = 大小B
...
    """
)

# 用户输入
user_input = st.text_area("请在此输入数据：", height=400)

# 输入数据解析函数
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
                    current_project = {"项目名": name, "总数据量 (MB)": size, "模块": []}
                else:
                    current_project["模块"].append({"模块名": name, "大小 (MB)": size})
            except ValueError:
                st.error("数据格式无效，请检查格式。")
                return []

        elif "-------------------" in line:
            if current_project:
                projects.append(current_project)
                current_project = None

    if current_project:
        projects.append(current_project)

    return projects

# 小数据合并为“其他”函数
def group_small_data(data, threshold=5):
    total_size = data["大小 (MB)"].sum()
    grouped_data = data[data["大小 (MB)"] >= (total_size * threshold / 100)]
    small_data = data[data["大小 (MB)"] < (total_size * threshold / 100)]

    other_details = ""
    if not small_data.empty:
        other_details = ", ".join(small_data["项目名"])
        other_row = pd.DataFrame({
            "项目名": ["其他"],
            "大小 (MB)": [small_data["大小 (MB)"].sum()]
        })
        grouped_data = pd.concat([grouped_data, other_row], ignore_index=True)

    return grouped_data, other_details

# 数据解析与可视化
if user_input:
    projects = parse_input_data(user_input)

    if projects:
        st.write("## 数据概览")
        all_modules = [{"项目名": p["项目名"], "模块名": m["模块名"], "大小 (MB)": m["大小 (MB)"]}
                       for p in projects for m in p["模块"]]
        df_modules = pd.DataFrame(all_modules)

        # 显示原始数据
        st.write("### 原始数据")
        st.dataframe(df_modules)

        # 统计信息
        st.write("### 数据统计信息")
        st.write(f"平均值: {df_modules['大小 (MB)'].mean():.2f} MB")
        st.write(f"中位数: {df_modules['大小 (MB)'].median():.2f} MB")
        st.write(f"众数: {df_modules['大小 (MB)'].mode()[0]:.2f} MB")
        st.write(f"标准差: {df_modules['大小 (MB)'].std():.2f} MB")
        st.write(f"最大值: {df_modules['大小 (MB)'].max():.2f} MB")
        st.write(f"最小值: {df_modules['大小 (MB)'].min():.2f} MB")

        # 直方图 + 正态分布 + 实际分布
        st.write("### 模块数据大小的分布（直方图 + 正态分布 + 实际分布）")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df_modules["大小 (MB)"], bins=20, kde=False, color="blue", stat="density", ax=ax, label="数据分布")
        mean = df_modules["大小 (MB)"].mean()
        std_dev = df_modules["大小 (MB)"].std()
        x = np.linspace(df_modules["大小 (MB)"].min(), df_modules["大小 (MB)"].max(), 100)
        normal_dist = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        ax.plot(x, normal_dist, color="red", label="正态分布")
        sns.kdeplot(df_modules["大小 (MB)"], color="purple", linewidth=2, label="核密度估计", ax=ax)
        y = np.full(len(df_modules["大小 (MB)"]), -0.02)
        ax.scatter(df_modules["大小 (MB)"], y, color="green", alpha=0.6, label="数据点", s=15)

        plt.xlabel("模块大小 (MB)", fontproperties=font_prop)
        plt.ylabel("密度", fontproperties=font_prop)
        plt.title("模块大小分布与正态分布对比", fontproperties=font_prop)
        plt.legend(prop=font_prop)

        st.pyplot(fig)

        # 箱线图
        st.write("### 各项目的大小分布（箱线图）")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_modules, x="项目名", y="大小 (MB)", ax=ax, palette="Set3", )
        plt.xticks(rotation=45)
        plt.title("各项目大小分布", fontproperties=font_prop)
        st.pyplot(fig)

        # 累积分布函数（CDF）
        st.write("### 累积分布函数（CDF）")
        sorted_sizes = np.sort(df_modules["大小 (MB)"])
        cdf = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sorted_sizes, cdf, marker=".", linestyle="none", color="orange")
        plt.xlabel("模块大小 (MB)", fontproperties=font_prop)
        plt.ylabel("累积比例", fontproperties=font_prop)
        plt.title("累积分布函数（CDF）", fontproperties=font_prop)
        st.pyplot(fig)

        # 顶级模块排名
        st.write("### 顶级模块排名")
        top_modules = df_modules.nlargest(5, "大小 (MB)")
        st.write("前 5 个模块：")
        st.table(top_modules)

        # 各项目的模块大小热图
        st.write("### 各项目的模块大小热图")
        try:
            pivot_table = df_modules.pivot_table(
                index="模块名",
                columns="项目名",
                values="大小 (MB)",
                fill_value=0
            )
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            plt.title("模块大小热图", fontproperties=font_prop)
            plt.xticks(fontproperties=font_prop)
            plt.yticks(fontproperties=font_prop)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"热图生成时发生错误: {e}")

        # 总体饼图
        st.write("### 数据比例（按项目）")
        try:
            project_sizes = pd.DataFrame({
                "项目名": [p["项目名"] for p in projects],
                "大小 (MB)": [p["总数据量 (MB)"] for p in projects]
            })

            project_sizes, other_details = group_small_data(project_sizes, threshold=5)

            fig, ax = plt.subplots(figsize=(10, 10))
            wedges, texts, autotexts = ax.pie(
                project_sizes["大小 (MB)"],
                labels=project_sizes["项目名"],
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("Set2", len(project_sizes)),
                wedgeprops={"edgecolor": "black", "linewidth": 0.5},
            )

            plt.title("各项目的数据比例", fontsize=16)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"饼图生成时发生错误: {e}")

else:
    st.info("请输入数据。")
