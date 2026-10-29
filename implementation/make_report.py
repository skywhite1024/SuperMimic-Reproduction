"""Create an evidence-backed local report from the verified downloaded bundle."""
from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    report = root.parent / "report"
    report.mkdir(exist_ok=True)
    evaluation = json.loads((root / "runs/baseline-evaluation.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "result-summary.json").read_text(encoding="utf-8"))
    proof = json.loads((root / "runs/baseline-seed0/update-proof-012000.json").read_text(encoding="utf-8"))
    assert summary["status"] == "baseline_complete" and proof["passed"]
    assert len(evaluation["per_clip"]) == 40
    reference_status="达到约 0.82 且零失败的参考目标" if summary["reference_target"]["met"] else "未达到约 0.82 且零失败的参考目标"
    failed_names=", ".join(html.escape(name) for name in evaluation["failed"]) or "无"
    failure_rows="".join(f'<tr><td>{html.escape(name)}</td><td>{evaluation["per_clip"][name]["steps"]*0.02:.2f}</td><td>{evaluation["per_clip"][name]["full_clip_steps"]*0.02:.2f}</td></tr>' for name in evaluation["failed"])
    rows = list(csv.DictReader((root / "runs/baseline-seed0/train_log.csv").open()))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(11,3.5), constrained_layout=True)
    iteration = [int(row["iteration"]) for row in rows]
    for key,label in (("ep_ret_norm","Training episode return"),("sim_rew","Simulation reward")):
        axes[0].plot(iteration,[float(row[key]) for row in rows],label=label,lw=1.4)
    axes[0].set(xlabel="Training iteration",ylabel="Logged value",title="Training logs (not independent evaluation)")
    axes[0].legend(fontsize=8)
    axes[1].plot(iteration,[float(row["wm_loss"]) for row in rows],lw=1.4,color="#c56c36")
    axes[1].set(xlabel="Training iteration",ylabel="World-model loss",yscale="log",title="World-model optimization")
    fig.savefig(report/"training.png",dpi=180)
    plt.close(fig)
    names = list(evaluation["per_clip"])
    clips = [evaluation["per_clip"][name] for name in names]
    fig,ax=plt.subplots(figsize=(9,10),layout="constrained")
    ax.barh(names,[x["norm_return"] for x in clips],color=["#b94f43" if x["failed"] else "#3d7d9b" for x in clips])
    ax.invert_yaxis()
    ax.set(xlabel="Full-clip normalized return",title="All 40 clips; red indicates early termination")
    ax.tick_params(axis="y",labelsize=7)
    fig.savefig(report/"per-clip.png",dpi=180)
    plt.close(fig)
    cells=[]
    for name,row in zip(names,clips):
        cells.append("<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in (
            name,f'{row["norm_return"]:.6f}',row["steps"],row["full_clip_steps"],f'{row["normalization_denominator"]:.1f}',"是" if row["failed"] else "否"))+"</tr>")
    resources=[]
    for name,row in summary["stages"].items():
        resources.append(f'<tr><td>{html.escape(name)}</td><td>{row["elapsed_seconds"]:.1f}</td>'
                         f'<td>{row["sampled_host_peak_bytes"]/1024**3:.2f}</td>'
                         f'<td>{row["sampled_gpu_peak_mib"]/1024:.2f}</td></tr>')
    video_html="".join(f'<figure><figcaption>{html.escape(p.stem)}</figcaption><video controls preload="metadata" src="../downloaded/runs/videos/{p.name}"></video></figure>' for p in sorted((root/"runs/videos").glob("*.mp4")))
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SuperMimic G1 LAFAN1 复现记录</title><style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#203440}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;font-size:14px;width:100%}}td,th{{border-bottom:1px solid #d9e2e7;padding:8px;text-align:left}}img,video{{max-width:100%;height:auto}}.stats{{padding:20px;background:#edf5f8;border-radius:10px}}code{{overflow-wrap:anywhere}}figure{{margin:20px 0}}a{{color:#176483}}</style>
<h1>SuperMimic · G1 · LAFAN1</h1>
<p>单张 H100 的社区实现基线，seed 0，40 条动作，12000 次迭代。</p>
<div class="stats"><b>独立评估宏平均：{evaluation["norm_return"]:.6f}</b>　帧数加权：{evaluation["norm_return_framew"]:.6f}　失败：{evaluation["n_failed"]}/40</div>
<p><b>{reference_status}。</b>这是 seed 0 的单次结果，归一化回报不是成功率。提前失败动作：{failed_names}。</p>
<p>这是 Motphys/Uni-SuperMimic 的社区基线。其单帧本体观测为 99 维，论文表格为 93 维；依赖锁定 UniLab 1.5.1。结果不代表原论文全部实验的严格复现。0.82 为参考目标，不能用训练日志替代独立评估。</p>
<p>社区 README 的参考结果使用 UniLab 1.5.0，本次遵循所固定源码提交的依赖锁，实际为 1.5.1。版本差异是待查因素；本轮没有对照实验，不能据此认定它造成分差。</p>
<h2>训练与恢复验收</h2><p>有限验收先执行 0→5 次迭代，退出后加载完整检查点并执行 5→8。世界模型、策略和 critic 的参数及优化器均通过实际更新检查；恢复时模型状态和优化器计数一致。回放缓冲区重新预热，因此不声称 RNG 与回放逐位连续。正式基线从独立的新目录开始。</p>
<p>正式训练保留上游 4096 环境、BF16、32 步想象时域和网络宽度。训练实现 commit：<code>faf7818e8abac3ec6abb8f00e6e904d3577f800b</code>。数据 revision：<code>ce1572906efe6157840e8474d5a0d7aa87481e74</code>。</p>
<img src="training.png" alt="训练曲线">
<h2>独立评估</h2><p>评估对象仍是训练使用的同一 40 条动作库；独立指单独进程和固定评估设置，不是未见动作的泛化测试。每条动作从第零帧开始，确定性策略，无观测/动作噪声，无域随机化，摩擦系数 1.0。提前失败仍按完整重采样动作长度减一乘以 0.1 归一化。宏平均与帧数加权平均分别报告，全部失败保留。</p>
<img src="per-clip.png" alt="40条动作评估"><table id="clip-results"><thead><tr><th>动作</th><th>归一化回报</th><th>执行步数</th><th>完整步数</th><th>分母</th><th>失败</th></tr></thead><tbody>{''.join(cells)}</tbody></table>
<h2>提前终止记录</h2><table><thead><tr><th>动作</th><th>终止时刻 / s</th><th>完整时长 / s</th></tr></thead><tbody>{failure_rows}</tbody></table><p>时长按实际控制步长 0.02s 换算。终止来自上游高度或姿态偏差判据；本轮未记录逐分支触发原因，不能把每条失败归因于某一个判据。没有删除失败样本，也没有改用存活长度缩小归一化分母。</p>
<h2>资源与时间</h2><table><thead><tr><th>阶段</th><th>子进程耗时 / s</th><th>采样主机峰值 / GiB</th><th>采样显存峰值 / GiB</th></tr></thead><tbody>{''.join(resources)}</tbody></table>
<p>资源每 5 秒采样，峰值不是连续测量；主机内存含 cgroup 文件缓存。PyTorch 分配器峰值：已分配 {proof["cuda_peak_allocated_bytes"]/1024**3:.2f}GiB，已保留 {proof["cuda_peak_reserved_bytes"]/1024**3:.2f}GiB，不包含 Warp 等非 PyTorch 分配。首个模型更新 {proof["first_world_model_update_seconds"]:.2f}s，首个策略更新 {proof["first_policy_update_seconds"]:.2f}s；启用编译时这些数值包含编译及更新，不能当作纯编译时间。排队、镜像和系统依赖准备不计入这些子进程时长。平台终态与 CCI/DNAT 释放证据见本地 execution.json。</p>
<h2>动作视频</h2><p>固定展示跳跃与起身（各 15 秒），再展示首个失败动作 dance2_subject4 的终止前片段（10 秒）。策略从第零帧滚动，再截取失败附近片段；没有在截取起点重新初始化。实色机器人为策略，蓝色半透明机器人为参考。三段视频共 1000 帧已全部解码，视频仅用于观察运动，不能替代数值评估。</p>{video_html}
<h2>资源释放</h2><p>本轮正式 GPU 作业于 2026-10-25 02:46:31 +08 显示成功。文件回收完成后，复用的 CPU CCI 已停止、0/1、实例列表为空；DNAT 已解绑，原启动配置已逐字恢复。自动停止 helper 仍不可用，本轮使用值守和控制台停止。云端结果摘要生成早于资源释放，其中的 false 保留原值；最终状态另存于 <a href="../final-resource-verification.json">最终资源核验记录</a>，不改写冻结结果。</p>
<details><summary>平台终态截图</summary><figure><figcaption>正式 ACP 成功</figcaption><img src="../evidence/resource-final-acp.jpg" alt="ACP 成功"></figure><figure><figcaption>CPU CCI 已停止</figcaption><img src="../evidence/resource-final-cci.jpg" alt="CCI 已停止，实例为空"></figure><figure><figcaption>DNAT 未绑定</figcaption><img src="../evidence/resource-final-dnat.jpg" alt="DNAT 未绑定"></figure></details>
<h2>可复查文件</h2><p><a href="../downloaded/runs/baseline-evaluation.json">完整评估 JSON</a> · <a href="../downloaded/result-manifest.json">逐文件哈希清单</a> · <a href="../local-transfer-verification.json">下载与逐文件校验</a> · <a href="../video-audit.json">视频解码验收</a> · <a href="../checkpoint-audit.json">检查点验收</a> · <a href="../execution.json">平台执行记录</a> · <a href="../README.md">复现说明</a></p><p>评估检查点 SHA256：<code>{evaluation["checkpoint_sha256"]}</code></p></html>'''
    (report/"supermimic-report.html").write_text(content,encoding="utf-8")
    print(report/"supermimic-report.html")


if __name__ == "__main__":
    main()
