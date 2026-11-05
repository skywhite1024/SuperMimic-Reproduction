"""Build the local, evidence-led report of the finite 1031 experiment."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name, default=None):
    path = ROOT/name
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def library(mode, model, device, variant):
    folder = f'runs/{mode}'
    summary = read(folder+'/summary.json')
    if summary is None:
        return None
    combined = read(folder+'/combined-assessment.json', {})
    tracking = read(folder+'/tracking-assessment.json', {})
    joined = {x['clip']: x for x in combined.get('clips', [])}
    quality = {x['clip']: x for x in tracking.get('clips', [])}
    rows = []
    for original in summary['clips']:
        name = original['clip']
        strict, detail = joined.get(name, {}), quality.get(name, original)
        rows.append({**original, 'strict': strict.get('combined_engineering_passed'),
                     'tracking_passed': strict.get('tracking_quality_passed'),
                     'contact_passed': strict.get('contact_review_passed'),
                     'tracking_failed_checks': strict.get('tracking_failed_checks', []),
                     'contact_checks': strict.get('contact_checks', {}),
                     'inversion_frame_coverage': detail.get('inversion_frame_coverage'),
                     'unexpected_trunk_contact_samples': strict.get('unexpected_trunk_contact_samples'),
                     'actual_min_after_initial_100ms_m': strict.get('actual_min_after_initial_100ms_m')})
    return {'mode': mode, 'model': model, 'device': device, 'variant': variant,
            'groups': summary['groups'], 'strict_new': combined.get('combined_engineering_passed'),
            'checkpoint_sha256': summary['checkpoint_sha256'], 'rows': rows,
            'summary_file': folder+'/summary.json', 'strict_file': folder+'/combined-assessment.json'}


def sequences(mode, model, device, variant):
    folder = f'runs/{mode}'
    assessed = read(folder+'/continuous-assessment.json')
    contact = read(folder+'/continuous-contact-review.json')
    if assessed is None or contact is None:
        return None
    contacts = {x['seed']: x for x in contact['trials']}
    rows = []
    for trial in assessed['trials']:
        physical = trial['original_trial']
        c = contacts[trial['seed']]
        rows.append({'seed': trial['seed'], 'seconds': physical['elapsed_seconds'],
                     'resets': physical['post_initialization_resets'],
                     'tracking_passed': trial['continuous_engineering_passed'],
                     'contact_passed': c['contact_review_passed'],
                     'joint_passed': trial['continuous_engineering_passed'] and c['contact_review_passed'],
                     'first_failure_s': physical['first_hard_failure_s'],
                     'failed_checks': [k for k, v in trial['checks'].items() if not v],
                     'failed_new_segments': [x['source'] for x in trial['new_full_segments'] if not x['passed']],
                     'minimum_contact_m': c['actual_min_after_initial_100ms_m'],
                     'unexpected_trunk_contact_samples': c['unexpected_trunk_contact_samples']})
    return {'mode': mode, 'model': model, 'device': device, 'variant': variant,
            'denominator': 3, 'tracking_passed': assessed['continuous_engineering_passed'],
            'contact_passed': contact['contact_review_passed'],
            'joint_passed': sum(row['joint_passed'] for row in rows), 'rows': rows,
            'assessment_file': folder+'/continuous-assessment.json',
            'contact_file': folder+'/continuous-contact-review.json'}


def main():
    libraries, continuous = [], []
    parent = {
        'cuda': ['retention-v1-parent-bound20-raw-cuda-new15',
                 'retention-v1-parent-bound20-corrected-cuda-all55',
                 'retention-v1-parent-bound20-raw-continuous-cuda',
                 'retention-v1-parent-bound20-corrected-continuous-cuda'],
        'cpu': ['stage3-actionbound20-raw-cpu-new15',
                'stage3-actionbound20-corrected-cpu-all55',
                'stage3-actionbound20-frozen-continuous-v1',
                'stage3-actionbound20-corrected-continuous-v1']}
    for model in ['parent', 'candidate']:
        for device in ['cuda', 'cpu']:
            modes = parent[device] if model == 'parent' else [
                f'qualified-expert-v2-raw-library-{device}',
                f'qualified-expert-v2-corrected-library-{device}',
                f'qualified-expert-v2-raw-sequences-{device}',
                f'qualified-expert-v2-corrected-sequences-{device}']
            for index, variant in enumerate(['raw', 'corrected']):
                item = library(modes[index], model, device, variant)
                if item:
                    libraries.append(item)
                item = sequences(modes[index+2], model, device, variant)
                if item:
                    continuous.append(item)
    videos = []
    for path in sorted((ROOT/'videos').glob('qualified-expert-v2-1031-*/*.json')):
        videos.append(json.loads(path.read_text()))
    parent_rows = next(x for x in libraries if (x['model'], x['device'], x['variant']) == ('parent', 'cuda', 'corrected'))['rows']
    candidate_rows = next(x for x in libraries if (x['model'], x['device'], x['variant']) == ('candidate', 'cuda', 'corrected'))['rows']
    before, after = {x['clip']: x for x in parent_rows}, {x['clip']: x for x in candidate_rows}
    changes = {
        'strict_new_gains': [k for k in after if after[k]['group'] == 'new' and after[k]['strict'] and not before[k]['strict']],
        'strict_new_losses': [k for k in after if after[k]['group'] == 'new' and before[k]['strict'] and not after[k]['strict']],
        'old_original_gains': [k for k in after if after[k]['group'] == 'lafan' and after[k]['passed_original_hard_gate'] and not before[k]['passed_original_hard_gate']],
        'old_original_losses': [k for k in after if after[k]['group'] == 'lafan' and before[k]['passed_original_hard_gate'] and not after[k]['passed_original_hard_gate']]}
    payload = {'updated_at': datetime.now(timezone.utc).astimezone().isoformat(),
               'libraries': libraries, 'continuous': continuous, 'videos': videos, 'CUDA_changes': changes,
               'CUDA_execution': read('evidence/qualified-expert-v2-CUDA-execution.json'),
               'CPU_execution': read('evidence/qualified-expert-v2-1031-CPU-assessment.json'),
               'verification': read('evidence/qualified-expert-v2-1031-cache-a-independent-verification.json'),
               'resources': read('evidence/qualified-expert-v2-1031-final-resource-verification.json'),
               'rendering': read('evidence/qualified-expert-v2-1031-video-rendering.json'),
               'supplementary_rendering': read('evidence/qualified-expert-v2-1031-supplementary-failure-videos.json'),
               'full_goal_complete': False, 'candidate_adopted': False}
    (ROOT/'evidence/qualified-expert-v2-1031-comparison.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    html = TEMPLATE.replace('PAYLOAD', json.dumps(payload, ensure_ascii=False).replace('</', '<\\/'))
    out = ROOT/'qualified-expert-v2-1031-report.html'
    out.write_text(html, encoding='utf8')
    print(json.dumps({'report': str(out), 'library_runs': len(libraries),
                      'continuous_runs': len(continuous), 'decoded_videos': len(videos),
                      'CPU_status': payload['CPU_execution']['status'], 'full_goal_complete': False}))


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SuperMimic G1 · 合格专家有限试验 · 2026-10-31</title>
<style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#18273c;background:#f3f6fa;max-width:1280px;margin:28px auto;padding:0 22px}h1{font-size:30px}h2{margin-top:34px}.card{background:white;border:1px solid #dbe3ec;border-radius:10px;padding:18px;margin:14px 0}.muted{color:#617187}.good{color:#116b49}.bad{color:#a23b25}.pending{color:#805f18}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #e2e8ef;text-align:left;vertical-align:top}th{background:#eaf0f7}select,button{padding:9px;border:1px solid #b8c8d9;border-radius:6px;background:white;margin:4px}a{color:#1267ac}.scroll{overflow:auto;max-height:650px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}video,img{max-width:100%;height:auto}code{overflow-wrap:anywhere}</style>
<h1>SuperMimic G1：合格专家有限试验</h1><p>2026-10-31 · 一个 101D 学生策略 · 新 15 段＋旧 40 段 · 原始／校准各 3×120 秒</p>
<div class="card"><strong>本轮已完成 1000 次训练更新，尚未达到完整复现目标；不采用新权重替换先前策略。</strong><p>CUDA 校准新动作原始硬门槛为 15/15，跟踪与接触联合严格验收为 10/15；旧动作原始门槛为 38/40。两组连续测试各有 1/3 同时通过跟踪与几何接触检查。新动作严格通过数未增加，旧动作相比匹配父策略退步，未满足预先声明的改进条件。</p><p class="muted">这是社区实现上的有限工程试验。严格验收要求完整动作、零重置、跟踪质量与几何接触均合格；旧 40 段单列原始门槛，不把它称为新动作严格质量。几何检查未测接触力。全部失败和失败后的完整轨迹均保留。</p></div>
<h2>CPU 与 CUDA 分开比较</h2><p>父策略与本轮均使用输出限幅 [-20,20]。CPU 采用从本轮完整检查点导出的相同策略张量；后端差异如实保留。</p><div class="card scroll" id="overview"></div>
<h2>动作库逐条结果</h2><div><select id="mode" aria-label="选择动作库评测"></select><select id="filter" aria-label="筛选动作"><option value="all">全部动作</option><option value="new">新15段</option><option value="lafan">旧40段</option><option value="failed">失败或未通过项</option></select><span id="row-count"></span></div><div class="card scroll" id="clips"></div>
<h2>固定连续测试：全部三条与失败尾段</h2><p>每条完整 120 秒，初始化后零重置。每条同时列出跟踪与几何接触结果，未通过的后续帧也保留在视频和轨迹中。</p><div class="card scroll" id="sequences"></div>
<h2>本轮变化与失败</h2><div class="card" id="changes"></div>
<h2>完整实际物理视频</h2><p>灰色为实际物理轨迹，绿色参考仅在显示时沿 Y 轴偏移 1.5 米，指标始终使用未偏移坐标。以下视频直接读取保留状态渲染；不调用教师替代学生，也不剪去失败尾段。</p><div class="grid" id="videos"></div>
<h2>训练、模型与资源</h2><div class="card" id="provenance"></div><details class="card"><summary>展开完整对比账本</summary><button id="export">导出 JSON</button><pre id="ledger"></pre></details><p class="muted" id="updated"></p>
<script>const D=PAYLOAD;const E=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const N=x=>x==null?'—':Number(x).toFixed(3);const M=x=>x==null?'<span class="pending">未评／待完成</span>':`<span class="${x?'good':'bad'}">${x?'通过':'失败'}</span>`;const label=x=>`${x.model==='parent'?'父策略30K':'本轮31K'} · ${x.device.toUpperCase()} · ${x.variant==='raw'?'原始':'校准'}`;const L=(m,d,v)=>D.libraries.find(x=>x.model===m&&x.device===d&&x.variant===v);const S=(m,d,v)=>D.continuous.find(x=>x.model===m&&x.device===d&&x.variant===v);
const rows=[];for(const m of ['parent','candidate'])for(const d of ['cuda','cpu']){const r=L(m,d,'raw'),c=L(m,d,'corrected'),sr=S(m,d,'raw'),sc=S(m,d,'corrected');rows.push(`<tr><td>${m==='parent'?'匹配父策略30K':'本轮候选31K'}</td><td>${d.toUpperCase()}</td><td>${c?.groups.new.passed??'待完成'}/15</td><td>${r?.strict_new??'待完成'}/15</td><td>${c?.strict_new??'待完成'}/15</td><td>${c?.groups.lafan?.passed??'待完成'}/40</td><td>${sr?.joint_passed??'待完成'}/3</td><td>${sc?.joint_passed??'待完成'}/3</td></tr>`)}document.querySelector('#overview').innerHTML='<table><thead><tr><th>模型</th><th>后端</th><th>校准新动作原始门槛</th><th>原始新动作严格</th><th>校准新动作严格</th><th>旧动作原始门槛</th><th>原始连续联合</th><th>校准连续联合</th></tr></thead><tbody>'+rows.join('')+'</tbody></table>';
const select=document.querySelector('#mode');select.innerHTML=D.libraries.map((x,i)=>`<option value="${i}">${E(label(x))}</option>`).join('');select.value=String(D.libraries.findIndex(x=>x.model==='candidate'&&x.device==='cuda'&&x.variant==='corrected'));
function clips(){const run=D.libraries[Number(select.value)],filter=document.querySelector('#filter').value;const rows=run.rows.filter(x=>filter==='all'||x.group===filter||(filter==='failed'&&(!x.passed_original_hard_gate||x.strict===false)));document.querySelector('#row-count').textContent=`显示 ${rows.length} / ${run.rows.length} 段`;document.querySelector('#clips').innerHTML='<table><thead><tr><th>动作／组</th><th>完整秒数／重置</th><th>原始门槛</th><th>严格质量</th><th>XY RMSE m</th><th>姿态／局部身体误差 m</th><th>翻转覆盖</th><th>未通过项</th></tr></thead><tbody>'+rows.map(x=>`<tr><td>${E(x.clip)}<br>${E(x.group)}</td><td>${N(x.elapsed_seconds)} / ${x.post_initialization_resets}</td><td>${M(x.passed_original_hard_gate)}</td><td>${x.group==='new'?M(x.strict):'旧动作原始合同'}</td><td>${N(x.root_xy_rmse_m)}</td><td>${N(x.body_global_mean_m)} / ${N(x.body_local_mean_m)}</td><td>${N(x.inversion_frame_coverage)}</td><td>${E([...x.tracking_failed_checks,...Object.entries(x.contact_checks).filter(([k,v])=>!v).map(([k])=>k)].join(', ')||'—')}</td></tr>`).join('')+`</tbody></table><p><a href="${E(run.summary_file)}">原始汇总</a> · <a href="${E(run.strict_file)}">严格验收原件</a></p>`}select.onchange=clips;document.querySelector('#filter').onchange=clips;clips();
document.querySelector('#sequences').innerHTML='<table><thead><tr><th>模型／后端／参考</th><th>seed</th><th>秒数／重置</th><th>跟踪</th><th>接触</th><th>联合</th><th>首次硬失败 s</th><th>未通过新动作段</th></tr></thead><tbody>'+D.continuous.flatMap(x=>x.rows.map(t=>`<tr><td>${E(label(x))}</td><td>${t.seed}</td><td>${t.seconds} / ${t.resets}</td><td>${M(t.tracking_passed)}</td><td>${M(t.contact_passed)}</td><td>${M(t.joint_passed)}</td><td>${N(t.first_failure_s)}</td><td>${E(t.failed_new_segments.join(', ')||'—')}</td></tr>`)).join('')+'</tbody></table>';
document.querySelector('#changes').innerHTML=`<p>CUDA 校准严格质量新增通过：${E(D.CUDA_changes.strict_new_gains.join(', ')||'无')}；丢失通过：${E(D.CUDA_changes.strict_new_losses.join(', ')||'无')}。</p><p>旧动作原始门槛新增通过：${E(D.CUDA_changes.old_original_gains.join(', ')||'无')}；丢失通过：${E(D.CUDA_changes.old_original_losses.join(', ')||'无')}。</p><p>这些是实测变化，不单独证明退化的因果来源。后续优先研究合格既有行为的保留约束，完整保留 15＋40 分母及冻结的连续测试。</p>`;
document.querySelector('#videos').innerHTML=D.videos.length?D.videos.map(v=>`<details class="card"><summary>${E(v.source_run)}<br>${E(v.clip)} · ${N(v.physical_window_seconds)}秒 · 全片解码通过</summary><video controls preload="none" src="${E(v.file)}"></video><p><a href="${E(v.file)}">打开完整视频</a> · ${v.frames}帧 / ${v.fps}fps</p></details>`).join(''):'<p class="pending">视频正在生成；尚未完成的视频不会列为通过解码。</p>';
const q=D.CUDA_execution,r=D.resources;document.querySelector('#provenance').innerHTML=`<p>训练更新 ${q.new_training_updates}，其中200次世界模型预热、800次联合更新；教师只参与训练。CUDA 原始／校准 88_08 教师控制在训练前通过；教师和归一化参数保持不变，部署只有一个101D学生。</p><p>新完整检查点 SHA256：<code>${E(q.checkpoint_sha256)}</code>。<a href="policy/qualified-expert-v2-h100-1000.pt">候选策略</a> · <a href="runs/unified-qualified-expert-v2-h100-1000/final-full.pt">完整检查点</a> · <a href="runs/unified-qualified-expert-v2-h100-1000/proof.json">训练与读回证明</a> · <a href="runs/unified-qualified-expert-v2-h100-1000/train_log_with_header.csv">训练CSV</a>。</p><p>结果归档 ${D.verification.archive_bytes} 字节，96个文件逐项核验。<a href="evidence/qualified-expert-v2-1031-cache-a-independent-verification.json">验证记录</a> · <a href="evidence/qualified-expert-v2-1031-comparison.json">完整对比JSON</a>。</p><p>GPU作业：${r.GPU_job_terminal?'已结束':'待核对'}；CPU容器：${r.CPU_CCI_stopped_and_empty?'已停止且实例为空':'待核对'}；DNAT：${r.DNAT_unbound?'已解绑':'待核对'}；原命令：${r.original_CCI_command_restored?'已恢复':'待核对'}。<a href="evidence/qualified-expert-v2-1031-final-resource-verification.json">资源原始记录</a>。</p><p>CPU评测：${E(D.CPU_execution.status)}；视频：${E(D.rendering?.status??'待运行')}。</p><details><summary>平台终态与CPU停止截图</summary><img src="evidence/qualified-expert-v4-cache-terminal-home.jpg" alt="GPU作业成功终态"><img src="evidence/cci-qualified-results-1031-final-stopped-empty.jpg" alt="CPU应用停止且无实例"></details>`;
document.querySelector('#ledger').textContent=JSON.stringify(D,null,2);document.querySelector('#updated').textContent='报告更新：'+D.updated_at+'。完整目标尚未完成；本轮候选未被采用。';document.querySelector('#export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(D,null,2)],{type:'application/json'}));a.download='qualified-expert-v2-1031-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>'''


if __name__ == '__main__':
    main()
