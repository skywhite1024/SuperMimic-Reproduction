"""Report all fixed repeats and observed divergence without promoting a teacher."""
import hashlib, html, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    result=json.loads((ROOT/'downloaded/teacher8808-repro1031c/recovered.json').read_text())
    analysis=result['analysis'];assert analysis['repetitions_completed']==3
    trials=analysis['trials'];repeat_pairs=[x for x in analysis['comparisons'] if x['a'].startswith('teacher8808-repro') and x['b'].startswith('teacher8808-repro')]
    same_first_action=all(x['first_action']['numeric_exact_equal'] for x in repeat_pairs)
    same_states=all(x['state']['numeric_exact_equal'] for x in repeat_pairs)
    runtime_equal=len({json.dumps({k:v for k,v in x['initial_policy'].items() if k not in ['mode']},sort_keys=True) for x in trials})==1
    earliest=[x['state']['first_differing_step'] for x in repeat_pairs if x['state']['first_differing_step'] is not None]
    conclusions={'reference_identical_across_history_and_fixed3':analysis['all_references_equal'],
                 'initial_state_identical_across_history_and_fixed3':analysis['all_initial_states_equal'],
                 'initial_history_identical_fixed3':analysis['history_hashes_equal_among_repeats'],
                 'initial_goal_identical_fixed3':analysis['goal_hashes_equal_among_repeats'],
                 'first_action_identical_fixed3':same_first_action,'full_state_trajectory_identical_fixed3':same_states,
                 'recorded_runtime_and_normalization_identical_fixed3':runtime_equal,
                 'earliest_state_divergence_among_fixed3':min(earliest) if earliest else None,
                 'raw_teacher_passes_fixed3':analysis['raw_teacher_passes'],'training_updates':0,
                 'candidate_adopted':False,'teacher_promoted':False,'full_goal_complete':False}
    (ROOT/'evidence/teacher8808-repro1031c-final-assessment.json').write_text(json.dumps(conclusions,indent=2)+'\n')
    previous=ROOT/'downloaded/qualified-expert-v2-1031-cache-a/extracted/runs/qualified-expert-v2-teacher-raw-cuda-8808'
    samples=json.loads((previous/'contact-samples.json').read_text())[0]['samples']
    prior_min=json.loads((previous/'contact-review.json').read_text())['clips'][0]['actual_min_after_initial_100ms_m']
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    axes[0].plot([x['time_s'] for x in samples],[x['actual_min_contact_distance_m']*1000 for x in samples],ls='--',c='black',alpha=.6,label='Previous V2 raw')
    for i,t in enumerate(trials):
        series=t['gate']['contact_distance_series'];axes[0].plot([x[0] for x in series],[x[1]*1000 for x in series],label=f'Fixed repeat {i}')
    axes[0].axhline(-50,color='firebrick',ls=':',label='Frozen -50 mm limit');axes[0].axvspan(0,.1,color='gray',alpha=.18)
    axes[0].set(xlabel='Physical time [s]',ylabel='Min. robot-ground distance [mm]',ylim=(-70,50),title='Near-ground view; shaded initial 100 ms excluded by the frozen gate')
    axes[0].legend(ncol=2,fontsize=8);axes[0].grid(alpha=.2)
    values=[-prior_min*1000,53.44258250133035]+[-t['gate']['actual_min_after_initial_100ms_m']*1000 for t in trials]
    axes[1].bar(['Previous V2','Oldw02 gate','Repeat 0','Repeat 1','Repeat 2'],values,color=['#8491a5','#cf715e','#5387a8','#5387a8','#5387a8'])
    axes[1].axhline(50,color='firebrick',ls=':');axes[1].set(ylabel='Max. penetration after 100 ms [mm]',title='Every retained result; teacher promotion remains disabled')
    for i,v in enumerate(values):axes[1].text(i,v+.5,f'{v:.3f}',ha='center',fontsize=9)
    axes[1].set_ylim(0,max(values)*1.2);axes[1].grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'evidence/teacher8808-repro1031c-contact-comparison.png',dpi=160)
    fig.savefig(ROOT/'evidence/teacher8808-repro1031c-contact-comparison.svg');plt.close(fig)
    rows=''.join(f'<tr><td>{i}</td><td>0</td><td>{-t["gate"]["actual_min_after_initial_100ms_m"]*1000:.3f}</td><td>{t["gate"]["tracking_passed"]}</td><td>{t["gate"]["unexpected_trunk_contact_samples"]}</td><td>{t["gate"]["passed"]}</td></tr>' for i,t in enumerate(trials))
    scope=('本次三次完整状态轨迹完全一致。历史任务之间仍存在差异，不能由这三次结果确定跨任务差异的原因。' if same_states else '本次固定三次之间仍出现状态轨迹差异。首个差异的时间及输入身份记录见下表；具体内核、求解器或隐藏状态来源尚未建立。')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>88_08 教师复现性诊断</title><style>body{{max-width:1050px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#18273c}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #c9d4df;padding:8px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}img{{max-width:100%}}.card{{padding:16px;background:#eef4fa;border-radius:8px}}</style>
<h1>88_08 教师：固定三次复现性诊断</h1><p>2026-10-31 · 3 次 raw · seed=0 · 独立进程 · 1 张 H100 · 0 训练更新</p><div class="card"><strong>固定三次教师门槛通过 {analysis['raw_teacher_passes']}/3；全部结果保留。</strong><p>{scope}</p><p>50 mm 接触门槛保持不变；没有提升教师资格或自动开始训练。完整新15＋旧40的统一策略目标仍未完成。</p></div>
<h2>逐次结果</h2><table><tr><th>重复</th><th>seed</th><th>初始100ms后最大穿地/mm</th><th>跟踪通过</th><th>意外躯干接触</th><th>联合通过</th></tr>{rows}</table>
<h2>输入与差异起点</h2><pre>{html.escape(json.dumps(conclusions,ensure_ascii=False,indent=2))}</pre><p>此前 V2 raw／校准在 88_08 使用逐元素相同的参考（Z位移0），初始状态和首动作相同；状态从第1步出现差异，动作从第16步出现差异。这是已观察的先后次序；不单独证明某个数值内核的因果来源。</p>
<img src="evidence/teacher8808-repro1031c-contact-comparison.png" alt="全部历史和固定三次接触结果"><p><a href="evidence/teacher8808-repro1031c-contact-comparison.svg">下载 SVG 图</a></p>
<h2>执行与证据</h2><p>实际作业 UID <code>768d1c2dbda94dc9b48ed42ab4f8fdc3</code>；程序总限时900秒，初始化120秒，单次评测180秒，零重启。初始化实际 {result['initialization_seconds']:.3f} 秒，程序总计 {result['elapsed_seconds']:.3f} 秒。3 次未因单个通过或失败而增加或筛除。</p><p><a href="downloaded/teacher8808-repro1031c/recovered.json">完整已核验结果</a> · <a href="teacher8808-repro1031c-plan.json">预先固定计划</a> · <a href="evidence/teacher8808-v2-v3-evidence-comparison1031.json">V2/V3 证据比较</a> · <a href="evidence/teacher8808-repro1031c-final-assessment.json">本次结论 JSON</a> · <a href="qualified-expert-v3-oldw02-1031-report.html">前一轮门槛报告</a> · <a href="qualified-expert-v2-1031-report.html">上一轮29段完整视频</a></p>
<details><summary>三次实际运行时和首次策略输入</summary><pre>{html.escape(json.dumps([x['initial_policy'] for x in trials],ensure_ascii=False,indent=2))}</pre></details></html>'''
    page=page.replace('<h2>执行与证据</h2>', '<h2>执行与证据</h2><p>平台已于20:18:46成功结束；原生清理记录无存活子进程。本轮没有启动CCI或修改DNAT。<a href="evidence/teacher8808-repro1031c-final-resource-verification.json">资源终态记录</a> · <a href="evidence/teacher8808-repro1031c-download-verification.json">结果下载核验</a></p><details><summary>平台终态截图</summary><img src="evidence/teacher8808-repro1031c-terminal.png" alt="教师诊断作业成功终态"></details>')
    report=ROOT/'teacher8808-repro1031c-report.html';report.write_text(page,encoding='utf8')
    print(json.dumps({'report':str(report),'conclusions':conclusions,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}))
if __name__=='__main__': main()
