"""Build a comparison only from downloaded, verified real experiment artifacts."""
from __future__ import annotations
import argparse
import csv
import gzip
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def esc(value):return html.escape(str(value))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    root=args.results.resolve(); local=root.parent
    assert load(local/'local-transfer-verification.json')['status']=='archive_and_every_file_verified'
    assert load(local/'artifact-audit.json')['status']=='downloaded_artifacts_audited'
    assert (local/'diagnostic-analysis.json').is_file()
    analysis=load(local/'diagnostic-analysis.json')
    drift_summary='，'.join(f'{name} {row["whole_clip_observed"]["root_position_error_m"]["max"]:.2f}m' for name,row in analysis['traces']['selected'].items())
    budget=load(local/'budget-ledger.json')
    report=local/'report';report.mkdir(exist_ok=True)
    experiment=load(root/'experiment.json');selection=load(root/'selection.json')
    baseline=load(root/'original-evaluation.json')
    index=load(root/'evaluations/index.json')
    primary=sorted([r for r in index if not r['confirmation']],key=lambda r:r['iteration'])
    results={row['iteration']:load(root/'evaluations'/f'{row["iteration"]:06d}.json') for row in primary}
    assert all(len(row['per_clip'])==40 for row in results.values())
    best_it=selection['best']['iteration'];final_it=selection['final']['iteration']
    best=results[best_it];final=results[final_it]
    confirmations=[load(root/'evaluations'/Path(row['result']).name) for row in index if row['confirmation']]
    auxiliary=root/'postprocess-20261026-a'
    auxiliary_used=(auxiliary/'evidence/postprocess.json').is_file()
    trace_root=auxiliary if auxiliary_used else root
    if auxiliary_used:assert load(auxiliary/'evidence/postprocess.json')['status']=='complete'
    confirmed=selection['target_confirmed'] and any(r['norm_return']>=.82 and r['n_failed']==0 for r in confirmations)
    assert confirmed==selection['target_confirmed']
    points=[(12000,baseline),*sorted(results.items())]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    xs=[i for i,_ in points]
    axes[0].plot(xs,[r['norm_return'] for _,r in points],'-o',label='Macro')
    axes[0].plot(xs,[r['norm_return_framew'] for _,r in points],'-s',label='Frame-weighted')
    axes[0].axhline(.82,ls='--',c='#aa6042',label='Community macro target')
    axes[0].set(xlabel='Total iteration',ylabel='Full-clip normalized return',title='Separate all-40 evaluations');axes[0].legend(fontsize=8)
    axes[1].plot(xs,[r['n_failed'] for _,r in points],'-o',c='#b6493f')
    axes[1].set(xlabel='Total iteration',ylabel='Early failures / 40',title='Original termination criteria unchanged')
    axes[1].set_yticks(range(max(r['n_failed'] for _,r in points)+1))
    for ax in axes:ax.set_xticks(xs)
    fig.savefig(report/'evaluation-trajectory.png',dpi=180);plt.close(fig)
    table=''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [i,f'{r["norm_return"]:.6f}',f'{r["norm_return_framew"]:.6f}',r['n_failed'],', '.join(r['failed']) or '无'])+'</tr>' for i,r in points)
    confirmation_rows=''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [row['iteration'],f'{result["norm_return"]:.6f}',f'{result["norm_return_framew"]:.6f}',result['n_failed'],result['checkpoint_sha256']])+'</tr>' for row,result in zip([r for r in index if r['confirmation']],confirmations))
    confirmation_section=('<h3>独立再次加载确认</h3><table><thead><tr><th>总迭代</th><th>宏平均</th><th>帧加权</th><th>失败数</th><th>检查点 SHA256</th></tr></thead><tbody>'+confirmation_rows+'</tbody></table>') if confirmation_rows else '<p>本次没有达到触发再次加载确认的双目标条件。</p>'
    perclip=''.join('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in [name,f'{old["norm_return"]:.6f}',f'{best["per_clip"][name]["norm_return"]:.6f}',f'{final["per_clip"][name]["norm_return"]:.6f}',best['per_clip'][name]['steps'],best['per_clip'][name]['full_clip_steps'],'是' if best['per_clip'][name]['failed'] else '否'])+'</tr>' for name,old in baseline['per_clip'].items())
    diagnostics=[]
    diagnostic_checks=[]
    for folder,label in [('parent-012000','12000'),('selected',str(best_it))]:
        receipt=load(trace_root/'diagnostics'/folder/'summary.json')
        if auxiliary_used:
            comparison=load(trace_root/'diagnostics'/folder/'primary-comparison.json')
            observed=receipt['observed_evaluation']
            diagnostic_checks.append('<tr>'+''.join(f'<td>{esc(value)}</td>' for value in [label,f'{observed["norm_return"]:.6f}',observed['n_failed'],f'{comparison["macro_delta"]:+.6f}',sum(row['strict_match_1e_minus5'] for row in comparison['per_clip']),max(abs(row['steps_delta']) for row in comparison['per_clip'])])+'</tr>')
    auxiliary_section=('<p>首次流水线在旧检查点的严格逐项回报复现检查处停止（dance1_subject1，阈值 1e−5）。训练和两次主验收已先完成；原失败状态、源码及日志均保留。随后独立辅助作业仅补齐轨迹和视频，没有训练或改写主结果。下表披露辅助轨迹与主评估的差异；诊断不是原数值轨迹的完全复现。</p><table><thead><tr><th>检查点</th><th>辅助宏平均</th><th>辅助失败数</th><th>相对主评估宏平均差</th><th>逐项严格匹配数 /40</th><th>最大步数差</th></tr></thead><tbody>'+''.join(diagnostic_checks)+'</tbody></table>') if auxiliary_used else '<p>诊断进程的全部 40 段回报、步数及终止标志通过了主评估结果一致性检查。</p>'
    for video_index,clip in enumerate(['dance2_subject4','fallAndGetUp1_subject5','fallAndGetUp2_subject3']):
        fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        fields=[('height_error_m','Height error / m',.25),('tilt_component_error','Tilt component error',.8),('root_position_error_m','Root position error / m',None),('root_orientation_error_rad','Root orientation error / rad',None),('joint_position_rmse_rad','Joint tracking RMSE / rad',None),('action_delta_rms','Action delta RMS',None)]
        for folder,label in [('parent-012000','12000 baseline'),('selected',f'{best_it} selected')]:
            with gzip.open(trace_root/'diagnostics'/folder/(clip+'.csv.gz'),'rt') as stream:trace=list(csv.DictReader(stream))
            for ax,(field,title,threshold) in zip(axes.flat,fields):
                ax.plot([float(r['time_seconds']) for r in trace],[float(r[field]) for r in trace],lw=1,label=label)
                ax.set(xlabel='Clip time / s',ylabel=title)
        for ax,(_,_,threshold) in zip(axes.flat,fields):
            if threshold is not None:ax.axhline(threshold,ls='--',c='black',lw=.8)
        axes[0,0].legend(fontsize=8);fig.suptitle(clip)
        fig.savefig(report/(clip+'-diagnostic.png'),dpi=160);plt.close(fig)
        old=load(trace_root/'diagnostics/parent-012000/summary.json')['clips'][clip]
        new=load(trace_root/'diagnostics/selected/summary.json')['clips'][clip]
        selected_summary=load(trace_root/'diagnostics/selected/summary.json')
        assert selected_summary.get('independent_trace_contract_checked',selected_summary['same_all40_rollout_metrics_reproduced'])
        gate=[]
        for key,label in [('height_gate','高度'),('tilt_gate','倾斜分量')]:
            if new['last_transition'][key]>.5:gate.append(label)
        caption=f'本次辅助轨迹中，12000 步策略终止/结束时刻 {old["seconds"]:.2f}s；新策略 {new["seconds"]:.2f}s。'
        caption+=('新策略提前终止，记录到的触发分支：'+('、'.join(gate) or '未记录')+'。') if new['failed'] else '新策略完成该动作的完整时长。'
        video_launch=load(trace_root/'runs'/f'video-{video_index}'/'launch.json')['command']
        video_start=float(video_launch[video_launch.index('--start')+1])
        video_seconds=float(video_launch[video_launch.index('--seconds')+1])
        caption+=f' 视频截取起点 {video_start:.2f}s，最多 {video_seconds:.0f}s，遇终止会缩短。'
        video_path=(trace_root/'videos'/(clip+'.mp4')).relative_to(root).as_posix()
        diagnostics.append(f'<h3>{esc(clip)}</h3><p>{caption}</p><img src="{clip}-diagnostic.png" alt="{clip} 误差轨迹"><video controls preload="metadata" src="../downloaded/{video_path}"></video>')
    stages=[]
    for p in sorted((root/'runs').glob('*/completion.json')):
        r=load(p);stages.append(f'<tr><td>{esc(p.parent.name)}</td><td>{r["elapsed_seconds"]:.2f}</td><td>{r["returncode"]}</td><td>{esc(r["stop_reason"])}</td></tr>')
    if auxiliary_used:
        for p in sorted((auxiliary/'runs').glob('*/completion.json')):
            r=load(p);stages.append(f'<tr><td>辅助 {esc(p.parent.name)}</td><td>{r["elapsed_seconds"]:.2f}</td><td>{r["returncode"]}</td><td>{esc(r["stop_reason"])}</td></tr>')
    release=load(local/'final-resource-verification.json') if (local/'final-resource-verification.json').is_file() else None
    released=bool(release and release.get('resource_release_verified'))
    headline='双目标已通过再次加载的完整评估确认' if confirmed else '本次仍未通过回报 ≥0.82 且零提前失败的双目标'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SuperMimic 受控续训对比</title>
<style>body{{font:16px/1.7 system-ui;max-width:1120px;margin:32px auto;padding:0 24px;color:#203440}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dbe3e8;text-align:left}}img,video{{max-width:100%;height:auto}}video{{display:block;margin:16px 0}}.score{{padding:20px;background:#edf4f7;border-radius:10px}}code{{overflow-wrap:anywhere}}a{{color:#176483}}</style>
<h1>SuperMimic · G1 LAFAN1 · 受控续训</h1><p>单 H100，seed 0，同一 40 段动作，冻结原评测标准。</p>
<div class="score"><b>{headline}。</b><br>选定检查点 {best_it}：宏平均 {best['norm_return']:.6f}，帧加权 {best['norm_return_framew']:.6f}，失败 {best['n_failed']}/40。<br>最终检查点 {final_it}：宏平均 {final['norm_return']:.6f}，帧加权 {final['norm_return_framew']:.6f}，失败 {final['n_failed']}/40。</div>
<p>16000 迭代达到双目标并经独立再次加载确认，按预定规则提前结束训练；未运行 20000 或 24000 迭代评估。图中连线只连接已测检查点。</p>
<p>这是此前约定的社区基线验收目标；不把 0/40 归为原论文明确给出的 LAFAN1 指标。归一化回报不是成功率。本次是单 seed 结果，同一训练动作库上的独立进程评估，不是未见动作泛化测试。</p><p>沿用社区实现的 99 维单帧本体观测和 UniLab 1.5.1；论文表格为 93 维，社区 README 参考环境为 UniLab 1.5.0。本轮没有更换这两项，也没有隔离实验来证明它们对分差的因果影响，因此仍按社区基线报告。</p>
<h2>受控改变与恢复</h2><p>从完整 12000 迭代 last.pt 延长训练，最多到 24000；观测、网络、奖励、终止、数据、依赖、物理及主评测脚本均冻结。模型与完整优化器 moment 状态在加载后通过哈希身份核验，前 5 次完整更新验收通过。学习率沿用绝对迭代位置的原余弦调度，lr=0.0002、lr_decay_iters=250000，未重新开始调度。</p>
<p>replay 重新预热至少 64 步，RNG、仿真状态及自适应起点统计不提供逐位恢复。阶段评测暂停同一个训练进程，不清空 replay。首次零 GPU 预检发现 tuple/JSON 数组的表示比较问题，修正了比较表示方式并保留旧版本；没有改模型或训练数值配置。</p>
<h2>全部阶段结果</h2><img src="evaluation-trajectory.png" alt="独立评估轨迹"><table><thead><tr><th>总迭代</th><th>宏平均</th><th>帧加权</th><th>失败数</th><th>失败动作</th></tr></thead><tbody>{table}</tbody></table>
{confirmation_section}
<p>“选定/最佳”按失败数最少、再按宏平均最高、再按迭代较新选择；最高宏平均的检查点为 {selection['highest_macro']['iteration']}。最终检查点单列保留。每段从首帧开始，确定性策略，无噪声及域随机化，摩擦 1.0；提前失败仍保留完整轨迹步数 ×0.1 的分母。失败均保留。</p>
<h2>40 段逐项比较</h2><table id="clip-comparison"><thead><tr><th>动作</th><th>12000</th><th>选定 {best_it}</th><th>最终 {final_it}</th><th>选定执行步数</th><th>完整步数</th><th>选定失败</th></tr></thead><tbody>{perclip}</tbody></table>
<h2>三条原失败动作的诊断与视频</h2>{auxiliary_section}<p>图中高度和倾斜虚线为冻结的终止阈值；倾斜量为 R[2,2] 相对参考的绝对差，单位不是弧度。关节位置误差与策略动作幅度/变化量分别记录，不能混用。视频展示实色策略与蓝色参考，先从首帧滚动，再截取失败附近或原失败时刻附近的片段。视频是单独的一次渲染轨迹，未断言与主评估或辅助数值轨迹逐帧相同。</p><p><a href="../diagnostic-analysis.json">终止分支、末秒/末十秒统计及全部 40 条变化的分析记录</a>仅作描述性证据，不单独证明学习失败的原因。</p>{''.join(diagnostics)}
<h2>耗时与资源</h2><table><thead><tr><th>阶段</th><th>程序耗时 / s</th><th>退出码</th><th>保护器停止原因</th></tr></thead><tbody>{''.join(stages)}</tbody></table><p>训练阶段耗时包含阶段评测暂停；各阶段评测耗时另见 evaluations/index.json，不能重复相加作为总训练时间。资源轨迹每 5 秒采样，含编译与初始化；采样峰值不是连续峰值。平台排队和镜像准备时间另见执行记录。</p>
<p>最终资源释放核验：{'已通过' if released else '尚未核验'}。云端结果摘要早于独立平台收尾核验，不改写冻结原文件。</p>
<p>本轮新增 GPU 作业按提交至终态窗口保守累计 {budget['conservative_GPU_hours']:.4f} GPU·小时，上限 6 小时；这是运行窗口估算，非平台账单口径。{budget['resource_trace_samples']} 份五秒资源采样中的主机内存峰值 {budget['sampled_host_peak_GiB']:.2f}GiB、GPU 显存峰值 {budget['sampled_gpu_peak_MiB']:.0f}MiB，未宣称连续峰值。详见 <a href="../budget-ledger.json">预算与采样账本</a>及 <a href="../final-resource-verification.json">资源收尾证据</a>。</p>
<p>零提前终止仅统计冻结高度/倾斜终止门限，跟踪仍有全局平移漂移：新策略三条辅助轨迹的根位置误差峰值为 {drift_summary}。视频相机跟随策略机器人，蓝色参考保留全局坐标，因此可能移出画面。</p>
<p><a href="../downloaded/evidence/full-final-readback.json">最终完整检查点 CPU 读回验收</a>核对模型、critic、完整优化器 moments 及绝对位置学习率与训练结束快照一致。读回在云端零 GPU CCI 执行；本机完成逐文件哈希、检查点 ZIP CRC 和全部视频帧解码，未执行本地模型推理。</p>
<h2>证据与检查点</h2><p><a href="../execution.json">平台执行记录</a> · <a href="../downloaded/selection.json">最终与最佳选择</a> · <a href="../downloaded/evidence/loaded-identity.json">完整恢复身份</a> · <a href="../downloaded/evidence/resume-acceptance.json">实际更新验收</a> · <a href="../downloaded/evaluations/index.json">全部阶段评估索引</a> · <a href="../local-transfer-verification.json">下载逐文件校验</a> · <a href="../README.md">实施说明</a></p><p>原基线 SHA256：<code>8770016fac48e4938203f77c8e2cd55b553171d2bc3134b7cbb40b2c53f6b1f1</code>。选定推理检查点：<a href="../downloaded/policy/ckpt_{best_it:06d}.pt">ckpt_{best_it:06d}.pt</a>；完整恢复检查点：<a href="../downloaded/policy/full_{best_it:06d}.pt">full_{best_it:06d}.pt</a>。原基线报告保持独立保存。</p></html>'''
    (report/'continuation-comparison.html').write_text(text,encoding='utf-8')
    print(report/'continuation-comparison.html')


if __name__=='__main__':main()
