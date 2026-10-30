"""Build an evidence-led report; missing runs remain visibly pending."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name, default=None):
    path = ROOT / name
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def main():
    modes = []
    for prefix, label in [('retention-v1-parent-bound20', 'Parent CUDA'), ('retention-v1', 'Candidate CUDA')]:
        modes += [(prefix + '-corrected-cuda-all55', label, 'calibrated'),
                  (prefix + '-raw-cuda-new15', label, 'raw'),
                  (prefix + '-raw-continuous-cuda', label, 'raw sequences'),
                  (prefix + '-corrected-continuous-cuda', label, 'calibrated sequences')]
    modes += [('stage3-actionbound20-corrected-cpu-all55', 'Parent CPU', 'calibrated'),
              ('stage3-actionbound20-raw-cpu-new15', 'Parent CPU', 'raw'),
              ('stage3-actionbound20-frozen-continuous-v1', 'Parent CPU', 'raw sequences'),
              ('stage3-actionbound20-corrected-continuous-v1', 'Parent CPU', 'calibrated sequences'),
              ('retention-v1-corrected-cpu-all55', 'Candidate CPU', 'calibrated'),
              ('retention-v1-raw-cpu-new15', 'Candidate CPU', 'raw'),
              ('retention-v1-raw-continuous-cpu', 'Candidate CPU', 'raw sequences'),
              ('retention-v1-corrected-continuous-cpu', 'Candidate CPU', 'calibrated sequences')]
    runs = []
    for mode, label, variant in modes:
        summary = read(f'runs/{mode}/summary.json')
        continuous = summary is not None and 'trials' in summary
        quality = read(f'runs/{mode}/' + ('continuous-assessment.json' if continuous else 'combined-assessment.json'))
        contact = read(f'runs/{mode}/continuous-contact-review.json')
        if contact:
            contact = {**contact, 'trials': [{key: value for key, value in row.items() if key != 'all_pose_contact_samples'}
                                           for row in contact['trials']]}
        videos = [json.loads(path.read_text(encoding='utf8')) for path in sorted((ROOT / 'videos' / mode).glob('*.json'))]
        runs.append({'mode': mode, 'label': label, 'reference': variant, 'summary': summary,
                     'quality': quality, 'tracking': read(f'runs/{mode}/tracking-assessment.json'),
                     'contact': contact, 'videos': videos})
    evidence_names = ['retention-resume1030b-submission.json', 'retention-resume1030b-runtime-verification.json',
                      'gpu-retention-v1-1000.json', 'gpu-retention-v1-1000-resume1030b.json',
                      'retention-1030-dependency-preflight.json', 'cci-retention-recovery-1030-final-resource-verification.json',
                      'retention-resume1030b-final-resource-verification.json', 'retention-1030-physical-assessment.json',
                      'retention-1030-video-verification.json', 'local-transfer-verification-retention-v1-1000-attempt-a.json',
                      'local-transfer-verification-retention-v1-1000-resume-b.json', 'retention-parent-CPU-reuse.json',
                      'retention-resume1030b-training-proof-from-console.json',
                      'retention-1030-anchor-diagnostic-from-console.json', 'retention-1030-coverage-diagnostic-from-console.json',
                      'retention-1030-canonical-console-reconciliation.json', 'retention-1030-failure-diagnostic.json',
                      'retention-1030-independent-model-verification.json',
                      'retention-1030-next-diagnostic-plan.json',
                      'retention-1030-public-entry-validation.json',
                      'retention-1030a-log-export-limitation.json']
    ledger = {'updated_at': datetime.now(timezone.utc).isoformat(), 'full_goal_complete': False,
              'scope': {'student_policies': 1, 'new_clips': 15, 'old_clips': 40, 'sequences_per_reference_and_backend': 3,
                        'sequence_seconds': 120, 'resets_after_t0': 0, 'new_updates_max': 1000, 'wm_only_updates': 200,
                        'teachers_used_only_during_training': True, 'automatic_extension': False, 'BONES_route': 'paused'},
              'plan': read('retention-trial-v1-plan.json'), 'resume_plan': read('retention-resume-1030b-plan.json'),
              'execution': read('execution.json'), 'proof': read('runs/unified-retention-v1-h100-1000/proof.json'),
              'evidence': {name: read('evidence/' + name) for name in evidence_names}, 'runs': runs,
              'figures': [f'figures/{name}.png' for name in ['retention-1030-training', 'retention-1030-new15-comparison']
                          if (ROOT / f'figures/{name}.png').exists()],
              'diagnostic_images': [{'file': 'evidence/' + name, 'caption': caption} for name, caption in [
                  ('retention-1030-candidate-85_07-at-5p72s.jpg', '85_07在5.72秒：原门槛和跟踪通过，几何接触仍失败；5.70–5.76秒共4次参考之外的骨盆触地。'),
                  ('retention-1030-candidate-continuous-seed0-at-9p68s.jpg', '校准CUDA seed0在9.68秒，邻近9.66秒XY误差峰值1.281米：完整120秒原门槛通过，严格跟踪仍失败。绿色参考仅作+1.5米Y显示平移，指标使用未平移世界坐标。')]
                  if (ROOT / 'evidence' / name).exists()]}
    physical = ledger['evidence']['retention-1030-physical-assessment.json']
    video = ledger['evidence']['retention-1030-video-verification.json']
    if physical and video:
        ledger['full_goal_complete'] = physical['raw_reference_goal_complete'] and not physical['video_generation_pending']
    (ROOT / 'retention-1030-status.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    payload = json.dumps(ledger, ensure_ascii=False).replace('</', '<\\/')
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SuperMimic retention v1：统一55动作受控试验</title>
<style>*{box-sizing:border-box}body{margin:0;color:#172b3d;background:#f2f5f8;font:15px/1.65 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:28px;line-height:1.35}h2{margin-top:30px}.card{background:white;border:1px solid #d5e0e8;padding:18px;border-radius:10px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.grid .card{margin:0}.metric{font-size:28px;font-weight:650}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #dfe7ed;padding:9px;text-align:left;vertical-align:top}th{background:#e7eff6;white-space:nowrap}small,.muted{color:#516779}.pass{color:#116947}.fail{color:#a63527}.pending{color:#88620e}a{color:#08669e}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}select,input,button{font:inherit;padding:7px;margin:5px 5px 5px 0}summary{cursor:pointer}video{width:100%;max-width:900px}</style>
<main><h1>Retention v1：一个策略的55动作受控试验</h1>
<p><a href="unified-library-report.html">历史三阶段</a> · <a href="action-bound-ablation-report.html">同权重限幅对照</a> · <a href="retention-trial-v1-plan.json">冻结实验计划</a> · <a href="retention-resume-1030b-plan.json">有限恢复计划</a></p>
<div class="card"><strong>比较范围：</strong>新15条和旧40条保留完整分母；原始参考与既定高度校准单独列出；CPU与CUDA分别统计。每项评测只在t=0初始化，失败后仍保留完整物理尾段。最多1000次新更新，前200次只更新世界模型；教师仅用于训练，部署只有一个101D学生策略。输出限幅为±20，沿用固定奖励、参考、物理和门槛。</div>
<div id="status" class="card"></div><div id="metrics" class="grid"></div>
<h2>所有分组结果</h2><div class="scroll" id="groups"></div>
<h2>逐条验收与完整失败尾段</h2><p>原始门槛、跟踪质量和几何接触分别列出。旧40条按冻结回归门槛验收；新动作质量标准的旧库诊断不替代原回归标准。</p>
<div class="card"><select id="mode" aria-label="选择评测"></select><input id="search" aria-label="搜索动作" placeholder="搜索动作名称"><select id="filter" aria-label="筛选结果"><option value="all">全部动作</option><option value="new">新15条</option><option value="lafan">旧40条</option><option value="failed">失败与未通过项</option></select><span id="count"></span></div><div class="scroll" id="clips"></div>
<h2>全部固定连续测试</h2><p>每组3条、每条完整120秒，原始和校准参考分别评测。转场只改变参考，不能重置机器人；几何接触检查未测接触力。</p><div id="sequences"></div>
<h2>完整分母对比与训练曲线</h2><div id="figures"></div>
<h2>失败、训练暴露与转场诊断</h2><div id="diagnosis" class="card"></div>
<div id="diagnostic-images" class="grid"></div>
<div id="next-test" class="card"></div>
<h2>训练与回收证据</h2><div class="card" id="provenance"></div><div id="resources" class="card"></div>
<details class="card"><summary>完整账本与全部指标</summary><button id="download">导出账本JSON</button><pre id="ledger"></pre></details><p id="updated" class="muted"></p>
<script>const D=PAYLOAD;
const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=x=>x==null?'—':Number(x).toFixed(3);const mark=x=>x==null?'<span class="pending">待验收</span>':'<span class="'+(x?'pass':'fail')+'">'+(x?'通过':'失败')+'</span>';
const video=v=>v?'<details><summary>完整物理视频 · '+n(v.physical_window_seconds)+'s</summary><video controls preload="none" src="'+esc(v.file)+'"></video><br><a href="'+esc(v.file)+'">打开视频</a> · '+(v.complete_decode_passed?'全片解码通过':'待核验')+'</details>':'<span class="pending">视频待生成</span>';
const physical=D.evidence['retention-1030-physical-assessment.json'];const trial=D.evidence['gpu-retention-v1-1000-resume1030b.json'];
const diagnostic=D.evidence['retention-1030-failure-diagnostic.json'];
document.querySelector('#diagnostic-images').innerHTML=(D.diagnostic_images??[]).map(x=>'<figure class="card"><img style="width:100%;height:auto" src="'+esc(x.file)+'" alt="'+esc(x.caption)+'"><figcaption>'+esc(x.caption)+'</figcaption></figure>').join('');
const nextTest=D.evidence['retention-1030-next-diagnostic-plan.json'];
document.querySelector('#next-test').innerHTML=nextTest?'<strong>下一项有限诊断：</strong>先完成本轮发布，再用0 GPU、0次训练更新检查stage2教师在相同输出限幅下的原始/校准完整新15动作。历史未限幅评测中，三条受保护动作严格通过数为0/3；这提示需要核实教师资格，不能替代匹配限幅的实测。当前计划状态：'+esc(nextTest.status)+'。<a href="evidence/retention-1030-next-diagnostic-plan.json">查看完整范围与截止条件</a>。不自动提交第二轮GPU训练。':'';
document.querySelector('#diagnosis').innerHTML=diagnostic?'<p>以下为完整记录的描述性分析，尚未建立训练退化的因果解释。教师只参与训练，部署始终为一个学生策略。</p><p>受保护动作实际参与策略更新的样本数：'+diagnostic.exposure.protected_clips.map(x=>esc(x.clip)+': '+x.PI_consumed).join('；')+'。采样规则按当时可用动作重新归一化；不能据此断言采样器违反计划。跨世界模型/策略池的最小可用起点计数也不能确定缺失发生的时刻和持续时间。</p><p>800条约束记录中，受保护新动作MSE前100条均值为'+n(diagnostic.teacher_anchor.statistics.protected_new_MSE.first_100_mean)+'，后100条为'+n(diagnostic.teacher_anchor.statistics.protected_new_MSE.last_100_mean)+'；样本上的教师损失下降并未保证闭环物理验收改善。</p><details><summary>四组匹配对比、全部24条连续测试的失败位置与接触记录</summary><pre>'+esc(JSON.stringify({libraries:diagnostic.libraries,sequences:diagnostic.sequences,inferences:diagnostic.inferences},null,2))+'</pre></details>':'完整匹配诊断待生成。';
document.querySelector('#status').innerHTML='<strong>整体目标：</strong>'+mark(D.full_goal_complete)+'；'+esc(D.execution?.stage)+'。'+(trial?'候选分类：'+esc(trial.classification??trial.status)+'。':'候选的完整程序收据待回收。')+' 运行状态和更新数不代表质量通过。';
document.querySelector('#metrics').innerHTML=[['实际新增更新',D.proof?D.proof.new_updates+' / 1000':'待完整证明'],['新动作分母','15'],['旧库分母','40'],['每组连续测试','3 × 120s'],['推理策略数','1'],['整体目标',D.full_goal_complete?'完成':'尚未完成']].map(([a,b])=>'<div class="card"><small>'+a+'</small><div class="metric">'+b+'</div></div>').join('');
document.querySelector('#groups').innerHTML='<table><thead><tr><th>模型/后端</th><th>参考</th><th>新15原门槛</th><th>新15跟踪+接触</th><th>旧40原门槛</th><th>旧库均分</th><th>连续严格</th></tr></thead><tbody>'+D.runs.map(r=>{const s=r.summary,q=r.quality;return '<tr><td>'+esc(r.label)+'</td><td>'+esc(r.reference)+'</td><td>'+(s?.groups?.new?s.groups.new.passed+'/15':'—')+'</td><td>'+(q?.new_denominator?q.combined_engineering_passed+'/15':'—')+'</td><td>'+(s?.groups?.lafan?s.groups.lafan.passed+'/40':'—')+'</td><td>'+n(s?.groups?.lafan?.macro_return_original)+'</td><td>'+(s?.trials?(q?q.continuous_engineering_passed+'/3':'待验收'):s?'—':'待回收')+'</td></tr>'}).join('')+'</tbody></table>';
const libraries=D.runs.filter(r=>!r.reference.includes('sequences'));const selector=document.querySelector('#mode');selector.innerHTML=libraries.map(r=>'<option value="'+esc(r.mode)+'">'+esc(r.label+' · '+r.reference)+'</option>').join('');selector.value='retention-v1-corrected-cuda-all55';
function render(){const r=libraries.find(x=>x.mode===selector.value),term=document.querySelector('#search').value.toLowerCase(),filter=document.querySelector('#filter').value;const quality=Object.fromEntries((r?.quality?.clips??[]).map(x=>[x.clip,x]));const tracking=Object.fromEntries((r?.tracking?.clips??[]).map(x=>[x.clip,x]));const videos=Object.fromEntries((r?.videos??[]).map(x=>[x.clip,x]));const rows=(r?.summary?.clips??[]).filter(x=>x.clip.toLowerCase().includes(term)&&(filter==='all'||filter===x.group||filter==='failed'&&(!x.passed_original_hard_gate||quality[x.clip]?.combined_engineering_passed===false)));document.querySelector('#count').textContent=r?.summary?'显示 '+rows.length+' / '+r.summary.clips.length+'；原始分母保持':'该项尚未回收';document.querySelector('#clips').innerHTML='<table><thead><tr><th>动作</th><th>组别/完整秒数</th><th>原门槛/均分</th><th>严格跟踪+接触</th><th>首次失败秒数</th><th>完整证据</th></tr></thead><tbody>'+rows.map(x=>'<tr><td>'+esc(x.clip)+'</td><td>'+esc(x.group)+' / '+n(x.full_clip_steps*.02)+'</td><td>'+mark(x.passed_original_hard_gate)+' / '+n(x.norm_return_original)+'</td><td>'+(x.group==='new'?mark(quality[x.clip]?.combined_engineering_passed):'见旧库回归门槛')+'</td><td>'+n(x.first_original_hard_failure_s)+'</td><td>'+video(videos[x.clip])+'<details><summary>全部指标和失败项</summary><pre>'+esc(JSON.stringify({physical:x,tracking:tracking[x.clip],combined:quality[x.clip]},null,2))+'</pre></details></td></tr>').join('')+'</tbody></table>'}
selector.onchange=render;document.querySelector('#search').oninput=render;document.querySelector('#filter').onchange=render;render();
document.querySelector('#sequences').innerHTML=D.runs.filter(r=>r.reference.includes('sequences')).map(r=>'<details class="card"><summary>'+esc(r.label+' · '+r.reference)+'：'+(r.quality?r.quality.continuous_engineering_passed+'/3 严格通过':'待验收')+'</summary>'+(r.summary?r.summary.trials.map(x=>'<div class="card">seed '+x.seed+' · 原门槛 '+mark(x.passed_original_hard_gate)+' · 首次失败 '+n(x.first_hard_failure_s)+'s · 重置 '+x.post_initialization_resets+video(r.videos.find(v=>v.clip==='seed-'+x.seed))+'<details><summary>完整窗口、转场与接触指标</summary><pre>'+esc(JSON.stringify({physical:x,quality:r.quality?.trials?.find(v=>v.seed===x.seed),contact:r.contact?.trials?.find(v=>v.seed===x.seed)},null,2))+'</pre></details></div>').join(''):'待回收')+'</details>').join('');
document.querySelector('#provenance').innerHTML='<p>首次作业在父模型接触审查时缺少SciPy，新增训练0次，原失败收据保留。恢复只补SciPy1.17.0和漏传的固定序列，不升级原七项依赖；复用已完成父55轨迹。恢复程序上限4200秒，与首次程序合计不超过5400秒。</p><p><a href="runs/unified-retention-v1-h100-1000/proof.json">实际更新、优化器、归一化和保存读回证明</a> · <a href="evidence/gpu-retention-v1-1000.json">首次失败收据</a> · <a href="evidence/retention-1030-dependency-preflight.json">CPU依赖与固定序列预检</a></p><p>原始/校准输入、CPU/CUDA后端、完整分母及所有失败尾段均保持。未通过预先冻结的改善条件时，候选记为失败；不自动追加训练。BONES全库路线继续暂停，本文不作论文等规模复现或真实机器人成功声明。</p><pre>'+esc(JSON.stringify(physical??{status:'物理验收尚未完成'},null,2))+'</pre>';
document.querySelector('#resources').innerHTML='<strong>资源释放：</strong><pre>'+esc(JSON.stringify(D.evidence['retention-resume1030b-final-resource-verification.json']??D.evidence['cci-retention-recovery-1030-final-resource-verification.json'],null,2))+'</pre>';
document.querySelector('#figures').innerHTML=D.figures.length?D.figures.map(file=>'<div class="card"><img src="'+esc(file)+'" style="width:100%;height:auto" alt="完整15条固定分母对比或实际1000次训练曲线"></div>').join(''):'<p class="pending">待完整训练日志和匹配评测回收后生成。</p>';
document.querySelector('#ledger').textContent=JSON.stringify(D,null,2);document.querySelector('#updated').textContent='证据快照：'+new Date(D.updated_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'});
document.querySelector('#download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(D,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='retention-1030-ledger.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></main></html>'''
    document = template.replace('PAYLOAD', payload)
    (ROOT / 'retention-1030-report.html').write_text(document, encoding='utf8')
    (ROOT / 'evidence/retention-1030-report-script.js').write_text(document.split('<script>', 1)[1].split('</script>', 1)[0], encoding='utf8')
    print(json.dumps({'report': 'retention-1030-report.html', 'expected_modes': len(runs),
                      'locally_verified_summaries': sum(row['summary'] is not None for row in runs),
                      'video_receipts': sum(len(row['videos']) for row in runs), 'full_goal_complete': ledger['full_goal_complete']}))


if __name__ == '__main__':
    main()
