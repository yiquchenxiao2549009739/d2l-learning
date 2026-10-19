"""完整 2.6 概率：抽样 -> 联合/条件概率 -> 贝叶斯 -> 期望与方差。

原教材 PDF 第 91-99 页，正文第 73-81 页，含节末练习。
检测参数沿用教材，改为虚构零件报警；只作概率演示。
现有 openmmlab 即可运行，无下载、无安装。
"""

import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
from torch.distributions import Categorical, Multinomial, Uniform

REPORT = ROOT / "reports" / "ch02"
FIGURES = REPORT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
torch.set_printoptions(precision=6, sci_mode=False)
SEED = 20261019
faces = torch.arange(1., 7., dtype=torch.float64)
fair_probs = torch.ones(6, dtype=torch.float64) / 6


def simulate_groups(m, n, seed=SEED):
    """每行记录 n 次独立投掷的六类计数；固定种子便于本机复现。"""
    torch.manual_seed(seed)
    counts = Multinomial(total_count=n, probs=fair_probs).sample((m,))
    cum_counts = counts.cumsum(dim=0)
    estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)
    return counts, cum_counts, estimates


def posterior(prior, likelihood_positive, false_positive):
    """事件 H 先验 + 给定 H 的报警概率 -> 报警后 H 的后验。"""
    evidence = prior * likelihood_positive + (1 - prior) * false_positive
    return prior * likelihood_positive / evidence


print("===== 1. 样本空间、事件与概率分布 =====")
odd_probability = fair_probs[[0, 2, 4]].sum()
print("公平骰子分布:", fair_probs)
print("概率总和:", fair_probs.sum().item(), "; 奇数事件概率:", odd_probability.item())
event_a = torch.tensor([False, False, False, True, True, True])  # 点数 >= 4。
event_b = torch.tensor([False, True, False, True, False, True])  # 偶数。
intersection = fair_probs[event_a & event_b].sum()
union = fair_probs[event_a | event_b].sum()
print("P(A)、P(B)、P(A且B)、P(A或B):",
      fair_probs[event_a].sum().item(), fair_probs[event_b].sum().item(),
      intersection.item(), union.item())

print("\n===== 2. 一次类别与多次类别计数不是同一个输出 =====")
torch.manual_seed(SEED)
single_label = Categorical(probs=fair_probs).sample()
single_counts = Multinomial(total_count=1, probs=fair_probs).sample()
ten_counts = Multinomial(total_count=10, probs=fair_probs).sample()
print("Categorical 的类别索引（0-5）:", single_label.item())
print("一次投掷的六类计数:", single_counts)
print("十次投掷的六类计数:", ten_counts, "; 合计:", ten_counts.sum().item())

print("\n===== 3. 累积频率：m 组，每组 n 次 =====")
m, n = 500, 10
counts, cum_counts, estimates = simulate_groups(m, n)
print("counts.shape:", tuple(counts.shape), "; 每行合计:", counts[0].sum().item())
print("前 3 组计数:\n", counts[:3])
print("总投掷次数:", int(cum_counts[-1].sum().item()))
print("最终经验频率:", estimates[-1])
print("频率和:", estimates[-1].sum().item())
group_records = []
for groups, per_group in [(500, 10), (50, 100), (1000, 100)]:
    batch, cumulative, frequency = simulate_groups(groups, per_group)
    max_error = (frequency[-1] - fair_probs).abs().max().item()
    group_records.append(dict(m=groups, n=per_group, total=groups*per_group,
                              max_abs_error=max_error))
    print("m=%d n=%d 总数=%d 最终最大绝对误差=%.6f" %
          (groups, per_group, groups*per_group, max_error))
with (REPORT / "2_6_group_comparison.csv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(group_records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(group_records)
with (REPORT / "2_6_fair_frequency.csv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["face", "count", "frequency", "true_probability"])
    for i in range(6):
        writer.writerow([i+1, int(cum_counts[-1, i].item()), estimates[-1, i].item(), 1/6])
print("大数定律不是每一步误差都减小，也不是下一次要补偿之前的结果。")

print("\n===== 4. 连续变量：密度与概率分开看 =====")
uniform = Uniform(torch.tensor(0., dtype=torch.float64), torch.tensor(0.5, dtype=torch.float64))
density_at_quarter = uniform.log_prob(torch.tensor(0.25, dtype=torch.float64)).exp()
interval_probability = (uniform.cdf(torch.tensor(0.2, dtype=torch.float64))
                        - uniform.cdf(torch.tensor(0.1, dtype=torch.float64)))
print("[0,0.5] 均匀分布的密度:", density_at_quarter.item())
print("落在 [0.1,0.2] 的概率:", interval_probability.item())
print("连续模型中，单点概率为 0；密度 2 不是概率 2。")

print("\n===== 5. 联合表 -> 边际 -> 条件概率 =====")
prior = 0.0015
sensitivity_1, false_positive_1 = 1.0, 0.01
joint = torch.tensor([[(1-prior)*(1-false_positive_1), (1-prior)*false_positive_1],
                      [prior*(1-sensitivity_1), prior*sensitivity_1]], dtype=torch.float64)
p_h = joint.sum(dim=1)
p_d = joint.sum(dim=0)
d_given_h = joint / p_h.reshape(-1, 1)
print("虚构零件联合表：行 H=0/1（合格/缺陷），列 D=0/1（不报警/报警）:\n", joint)
print("P(H):", p_h, "; P(D):", p_d)
print("P(D|H)，按 H 行归一化:\n", d_given_h)
post_first = posterior(prior, sensitivity_1, false_positive_1)
print("P(D=1|H=1):", sensitivity_1, "; P(H=1|D=1):", post_first)
print("一百万件中的期望：缺陷且报警 1500，合格却报警 9985。")

print("\n===== 6. 贝叶斯更新与额外独立性假设 =====")
sensitivity_2, false_positive_2 = 0.98, 0.03
post_both = posterior(prior, sensitivity_1*sensitivity_2, false_positive_1*false_positive_2)
post_sequential = posterior(post_first, sensitivity_2, false_positive_2)
post_same_independent = posterior(prior, sensitivity_1**2, false_positive_1**2)
print("两种报警均触发，给定 H 后条件独立时，后验:", post_both)
print("依次更新与一次联合计算一致:", post_sequential)
print("若两次一号检验也条件独立，则后验:", post_same_independent)
print("若只是完全重复同一个确定性结果，没有新证据，后验仍为:", post_first)

print("\n===== 7. 独立、互斥、条件独立是不同概念 =====")
two_dice_joint = fair_probs[:, None] * fair_probs[None, :]
print("两次独立投掷 P(6,6):", two_dice_joint[5, 5].item())
print("同一次投掷的 1 与 2 互斥，交集概率 0，但概率乘积为", 1/36)
coin_probabilities = torch.tensor([[0.8, 0.2], [0.2, 0.8]], dtype=torch.float64)
conditional_coin_joint = coin_probabilities[:, :, None] * coin_probabilities[:, None, :]
mixed_coin_joint = conditional_coin_joint.mean(dim=0)
print("先等概率选一枚硬币，给定硬币后两次独立；不知硬币时联合表:\n", mixed_coin_joint)
print("P(两次正面)=", mixed_coin_joint[1, 1].item(), "; 单独正面概率之积=0.25")

print("\n===== 8. 教材练习：概率界与马尔可夫链 =====")
p_a, p_b = 0.6, 0.7
intersection_bounds = (max(0, p_a+p_b-1), min(p_a, p_b))
union_bounds = (max(p_a, p_b), min(1, p_a+p_b))
print("已知 P(A)=0.6、P(B)=0.7，交集范围:", intersection_bounds, "; 并集范围:", union_bounds)
chain_prior = torch.tensor([0.6, 0.4], dtype=torch.float64)
transition_ab = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float64)
transition_bc = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64)
chain_joint = chain_prior[:, None, None] * transition_ab[:, :, None] * transition_bc[None, :, :]
p_c = chain_joint.sum(dim=(0, 1))
print("P(A,B,C)=P(A)P(B|A)P(C|B)，联合形状:", tuple(chain_joint.shape))
print("边际 P(C):", p_c)
print("P(C=1|A=0/1):", (transition_ab @ transition_bc)[:, 1])

print("\n===== 9. 期望、方差与样本统计 =====")
fair_mean = torch.dot(faces, fair_probs)
fair_second_moment = torch.dot(faces.square(), fair_probs)
fair_variance = torch.dot((faces-fair_mean).square(), fair_probs)
loaded_probs = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5], dtype=torch.float64)
loaded_mean = torch.dot(faces, loaded_probs)
loaded_variance = torch.dot((faces-loaded_mean).square(), loaded_probs)
concentrated_probs = torch.tensor([0., 0., 0.5, 0.5, 0., 0.], dtype=torch.float64)
concentrated_mean = torch.dot(faces, concentrated_probs)
concentrated_variance = torch.dot((faces-concentrated_mean).square(), concentrated_probs)
empirical_mean = torch.dot(faces, estimates[-1])
empirical_variance = torch.dot((faces-empirical_mean).square(), estimates[-1])
print("公平骰子期望 / 方差 / 标准差:", fair_mean.item(), fair_variance.item(), fair_variance.sqrt().item())
print("E[X²] / E[X]²:", fair_second_moment.item(), fair_mean.square().item())
print("偏骰期望 / 方差:", loaded_mean.item(), loaded_variance.item())
print("只落在 3、4 的分布：期望相同，方差不同:", concentrated_mean.item(), concentrated_variance.item())
print("模拟经验均值 / 经验方差:", empirical_mean.item(), empirical_variance.item())
print("对 1..6 等权计算 var(correction=0):", faces.var(correction=0).item())
print("注意：任意非均匀分布的方差要按概率加权。")

print("\n===== 10. 概率怎样接到以后要训练的分类模型 =====")
class_probs = torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64)
true_class = 1
negative_log_loss = -torch.log(class_probs[true_class])
print("三类概率:", class_probs, "; 真实类别索引:", true_class)
print("该类别的负对数概率:", negative_log_loss.item())
print("分类模型如何产生概率与训练，在后续分类章节正式学习。")

print("\n===== 11. 串起一个联合抽样与条件频率流程 =====")
torch.manual_seed(SEED + 1)
joint_counts = Multinomial(total_count=1000, probs=joint.reshape(-1)).sample((100,))
empirical_joint = (joint_counts.sum(dim=0) / joint_counts.sum()).reshape(2, 2)
empirical_posterior = empirical_joint[1, 1] / empirical_joint[:, 1].sum()
print("十万零件的经验联合表:\n", empirical_joint)
print("由经验表计算的报警后缺陷频率:", empirical_posterior.item())
print("对应理论后验:", post_first)


def make_figures():
    """绘图辅助区，数据来自本节概率计算。"""
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "savefig.facecolor": "white"})
    teal, orange, navy = "#147E83", "#D07632", "#173448"

    def save(fig, name):
        fig.savefig(FIGURES/name, dpi=180, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print("配图:", name)

    fig, ax = plt.subplots(figsize=(9.2, 3.25))
    total_throws = torch.arange(1, m+1) * n
    for i in range(6):
        ax.plot(total_throws.numpy(), estimates[:, i].numpy(), lw=1.15, label="点数 %d" % (i+1))
    ax.axhline(1/6, color=navy, ls="--", lw=1.2, label="真实概率 1/6")
    ax.set(xscale="log", xlabel="累计投掷次数（对数刻度）", ylabel="累计频率",
           title="同一条随机序列：频率逐渐稳定，但并非每一步都更接近")
    ax.legend(ncol=4, fontsize=8, loc="upper right", framealpha=1)
    ax.grid(alpha=0.16)
    fig.tight_layout()
    save(fig, "2_6_frequency_convergence.png")

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    bars = ax.bar(["报警前", "一号报警后", "两号均报警后"],
                  [prior, post_first, post_both], color=[navy, teal, orange], width=0.55)
    for bar, value in zip(bars, [prior, post_first, post_both]):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.025, "%.2f%%" % (100*value), ha="center", fontsize=11)
    ax.set(ylim=(0, 1), ylabel="缺陷概率", title="虚构零件报警：后验取决于先验与证据；双报警另需条件独立假设")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    save(fig, "2_6_bayes_update.png")

    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.set(xlim=(-0.2, 10.2), ylim=(-1, 3))
    ax.axis("off")
    nodes = [(3.1, 1.6, "C：选中的硬币\n正面概率 0.2 或 0.8"),
             (0.3, 0.0, "A：第一次结果"), (6.0, 0.0, "B：第二次结果")]
    for left, bottom, label in nodes:
        ax.add_patch(FancyBboxPatch((left,bottom),3.2,0.85,boxstyle="round,pad=0.04",fc="#F1F7F8",ec="#D7E2E9"))
        ax.text(left+1.6,bottom+0.425,label,ha="center",va="center",color=navy,fontsize=11)
    for end in [(1.9,0.95),(7.6,0.95)]:
        ax.annotate("",end,(4.7,1.55),arrowprops=dict(arrowstyle="->",color=teal,lw=1.8))
    ax.text(4.7,-0.55,"给定 C 后 A、B 独立；未知 C 时 P(A=1,B=1)=0.34，不等于 0.25",ha="center",color=orange,fontsize=10)
    save(fig, "2_6_conditional_independence.png")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    distributions = [(fair_probs, fair_mean, fair_variance, "公平骰子"),
                     (loaded_probs, loaded_mean, loaded_variance, "偏骰子"),
                     (concentrated_probs, concentrated_mean, concentrated_variance, "只落在 3、4")]
    for ax, (probs, mean, variance, title) in zip(axes, distributions):
        ax.bar(faces.numpy(), probs.numpy(), color=teal, width=0.65)
        ax.axvline(mean.item(), color=orange, ls="--", lw=1.5)
        ax.set(xticks=range(1,7), ylim=(0,0.6), xlabel="点数", title=title)
        ax.text(0.05,0.92,"均值 %.2f\n方差 %.3f" % (mean,variance),transform=ax.transAxes,
                va="top",fontsize=9,bbox=dict(facecolor="white",edgecolor="none",alpha=0.95))
    axes[0].set_ylabel("概率")
    fig.tight_layout(w_pad=1.8)
    save(fig, "2_6_mean_variance.png")


make_figures()
print("\n整节结束：完成综合练习的联合抽样流程，再一次反馈结果和三个解释。")
