"""Numerical recovery report, explicitly separate from pending large-artifact collection."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'specialist8807-1031d'


def main():
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:2])
    result = json.loads((ROOT/f'downloaded/{LABEL}/recovered.json').read_text())
    resource = json.loads((ROOT/f'evidence/{LABEL}-final-resource-verification.json').read_text())
    cpu = json.loads((ROOT/f'evidence/{LABEL}-parent-CPU-assessment.json').read_text())
    rows = [dict(row, role=role) for role in ['parent', 'candidate'] for row in result[role]]
    metrics = [('Root XY RMSE (m)', lambda x:x['tracking']['root_xy_rmse_m'], .35),
               ('Local body error (m)', lambda x:x['tracking']['body_local_mean_m'], .15),
               ('Root orientation error (deg)', lambda x:x['tracking']['root_orientation_geodesic_rmse_deg'], 35),
               ('Original normalized return', lambda x:x['tracking']['norm_return_original'], .65),
               ('Unexpected trunk samples', lambda x:x['unexpected_trunk_contact_samples'], 0),
               ('Penetration after 100ms (mm)', lambda x:max(0,-1000*x['actual_min_after_initial_100ms_m']), 50)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (title, measure, threshold) in zip(axes.flat, metrics):
        values = [[measure(row) for row in result[role]] for role in ['parent', 'candidate']]
        means = np.array([np.mean(v) for v in values])
        ranges = np.array([[means[i]-min(v) for i,v in enumerate(values)],
                           [max(v)-means[i] for i,v in enumerate(values)]])
        axis.bar(['16K parent','17K specialist'], means, yerr=ranges,
                 color=['#7b879a','#2d6fbb'], capsize=5)
        axis.axhline(threshold, color='#b8354b', linestyle='--', label='Frozen threshold')
        axis.set_title(title)
        axis.grid(axis='y', alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('CUDA: all six repetitions per model, means and complete range')
    picture = ROOT/f'evidence/{LABEL}-CUDA-numerical-comparison.png'
    fig.savefig(picture, dpi=160)
    fig.savefig(picture.with_suffix('.svg'))
    plt.close(fig)
    escape = lambda x: html.escape(str(x), quote=True)
    content = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>88_07 专项训练数值结果</title><style>body{font:16px/1.8 system-ui;background:#f4f7fb;color:#18283d;margin:0}main{max-width:1200px;padding:28px;margin:auto}section{padding:22px;background:white;border:1px solid #d8e1ed;border-radius:12px;margin:22px 0}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #d8e1ed}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}img,video{max-width:100%}.scroll{overflow-x:auto}.fail{color:#a52b3d}</style><main>',
        '<p>有限实验 1031d · 99D 独立专家 · 数值与资源终态已回收</p><h1>88_07 有姿态改善，严格验收未通过</h1>',
        '<section><p><strong>1000次新增更新已完成：400次世界模型预热＋600次联合学习，16K→17K。</strong></p><p>云端于2026年10月31日23:02:07成功结束，59阶段退出0，原生子进程组全部为空。上次22:44左右登录过期中断了助手的查看与回收，云端程序继续完成固定验收和两套55动作边界检查；本轮没有重提训练。</p>',
        '<p class="fail"><strong>严格目标验收：原始0/3→0/3，校准0/3→0/3。独立专家分支旧40硬门槛：40/40→0/40，候选不作为统一模型采用。</strong></p><p>6次候选目标轨迹均通过原始硬门槛，但跟踪质量和接触严格门槛仍失败。仅通过硬门槛不能代表动作质量合格。</p></section>',
        f'<section><h2>固定目标全部重复</h2><p>原始与高度校准参考逐字节相同，校准Z偏移0；标签保留，两组均为seed0、独立进程。每次126控制步／2.52秒，初始化后零重置、失败尾部全部保留。图中误差棒为全部6次的最小—最大范围。</p><img src="evidence/{picture.name}" alt="姿态改善但位移误差及严格门槛仍失败"><div class="scroll"><table><tr><th>模型／参考／次数</th><th>严格</th><th>原硬门槛</th><th>XY RMSE m</th><th>局部姿态 m</th><th>朝向误差 °</th><th>回报</th><th>穿地 mm</th><th>意外躯干接触</th></tr>']
    for x in rows:
        t=x['tracking']
        content.append(f'<tr><td>{escape(x["role"])} / {escape(x["reference_variant"])} / {x["repetition"]}</td><td>{x["strict_passed"]}</td><td>{t["passed_original_hard_gate"]}</td><td>{t["root_xy_rmse_m"]:.4f}</td><td>{t["body_local_mean_m"]:.4f}</td><td>{t["root_orientation_geodesic_rmse_deg"]:.2f}</td><td>{t["norm_return_original"]:.4f}</td><td>{max(0,-1000*x["actual_min_after_initial_100ms_m"]):.3f}</td><td>{x["unexpected_trunk_contact_samples"]}</td></tr>')
    content += ['</table></div><p>候选局部姿态误差约0.155–0.161m，门槛≤0.15m；XY RMSE约0.705–0.816m，门槛≤0.35m；每次仍有3个意外躯干接触样本，门槛0。接触是几何审查，未测量接触力。完整失败项见下方原始结果。</p></section>',
        '<section><h2>CUDA校准55：能力边界与遗忘</h2><table><tr><th>模型</th><th>新15原硬门槛</th><th>新15严格</th><th>旧40原硬门槛</th><th>旧40宏平均回报</th></tr>']
    for role in ['parent','candidate']:
        b=result['boundary'][role];groups=b['groups']
        content.append(f'<tr><td>{role}</td><td>{groups["new"]["passed"]}/15</td><td>{b["new_strict"]}/15</td><td>{groups["lafan"]["passed"]}/40</td><td>{groups["lafan"]["macro_return_original"]:.5f}</td></tr>')
    content += ['</table><p>这是独立99D专家分支的边界检查，不能当最终101D统一策略。原16K、30K统一模型及历史模型保留。最终新15＋旧40、raw/cal各3×120秒连续目标尚未完成，本轮没有运行连续验收。</p></section>',
        '<section><h2>已经回收与尚待回收</h2><p>已回收：115条完整控制台日志，93段结果重组，字节数与SHA256通过；12次CUDA目标评测全部数值、两套55边界全部逐动作原硬门槛指标、训练证明与资源终态。父模型本地CPU原始0/3、校准0/3，6段CPU视频已完整解码验证。</p><p>尚待回收：686,466,636字节AFS归档中的完整权重、训练CSV及CUDA物理轨迹。它们未完成本地逐文件验证；候选CPU匹配评估、CUDA视频和训练曲线尚未完成。此报告不把云端归档哈希冒充本地下载验证。</p><pre>'+escape(json.dumps(result['archive'],indent=2))+'</pre></section>',
        '<section><h2>下一项可检验假设，尚未执行</h2><p>优先检验全局位移观测是否不足：在原奖励、物理、参考和验收门槛不变的条件下，只增加全局XY误差观测，检查能否降低漂移。现有101D实现同时附加XY奖励，不能不加区分地复用后宣称只有观测改变。独立专家与旧库保留需要分别验收，本轮结果不支持直接提升为合格蒸馏教师。</p></section>',
        '<section><h2>全部CUDA数值与失败</h2><details><summary>12次固定目标评测</summary><pre>'+escape(json.dumps(rows,ensure_ascii=False,indent=2))+'</pre></details><details><summary>两套55动作逐条指标</summary><pre>'+escape(json.dumps(result['boundary'],ensure_ascii=False,indent=2))+'</pre></details><details><summary>训练与导出回读证明</summary><pre>'+escape(json.dumps(result['training_proof'],ensure_ascii=False,indent=2))+'</pre></details><details><summary>资源终态</summary><pre>'+escape(json.dumps(resource,ensure_ascii=False,indent=2))+'</pre></details><p><a href="downloaded/'+LABEL+'/recovered.json">原始重组结果JSON</a> · <a href="evidence/'+LABEL+'-console-recovery.json">日志完整性核验</a> · <a href="evidence/'+LABEL+'-terminal.png">真实平台终态</a></p></section></main></html>']
    report=ROOT/f'{LABEL}-numerical-report.html'
    report.write_text('\n'.join(content),encoding='utf8')
    validation={'status':'numerical_report_generated_partial_collection_explicit','report':report.name,
        'target_CUDA_rows':len(rows),'CUDA_boundary_rows':sum(len(x['summary_clips']) for x in result['boundary'].values()),
        'candidate_CPU_assessed':False,'CUDA_trajectories_downloaded':False,'model_archive_locally_verified':False,
        'full_goal_complete':False,'report_sha256':__import__('hashlib').sha256(report.read_bytes()).hexdigest()}
    (ROOT/f'evidence/{LABEL}-numerical-report-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation),flush=True)


if __name__ == '__main__':
    main()
