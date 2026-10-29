"""Reviewable same-weight CPU ablation, with all denominators and failed tails."""
import html
import json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path, default=None):
    target=ROOT/path
    return json.loads(target.read_text(encoding='utf8')) if target.exists() else default


def main():
    comparison=read('evidence/action-bound-matched-comparison.json')
    completion=read('evidence/action-bound-completion.json')
    modes=['unified-stage3-corrected-cpu-all55','stage3-actionbound20-corrected-cpu-all55',
           'stage3-raw-cpu-new15-control','stage3-actionbound20-raw-cpu-new15',
           'stage3-frozen-continuous-v1','stage3-actionbound20-frozen-continuous-v1',
           'stage3-corrected-continuous-v1','stage3-actionbound20-corrected-continuous-v1']
    runs=[]
    for mode in modes:
        summary=read(f'runs/{mode}/summary.json')
        if not summary:continue
        quality=read(f'runs/{mode}/combined-assessment.json') or read(f'runs/{mode}/continuous-assessment.json')
        contact=read(f'runs/{mode}/continuous-contact-review.json')
        if contact:
            # Keep large 20ms contact samples in the original evidence file; summaries retain all events/counts.
            contact={**contact,'trials':[{k:v for k,v in row.items() if k!='all_pose_contact_samples'} for row in contact['trials']]}
        videos=[]
        for path in sorted((ROOT/'videos'/mode).glob('*.json')):
            videos.append(read(str(path.relative_to(ROOT))))
        runs.append({'mode':mode,'summary':summary,'quality':quality,'contact':contact,'videos':videos})
    ledger={'updated_at':datetime.now(timezone.utc).isoformat(),'training_updates':0,
            'controller_variant':'action-output-bound-20-v1','device':'cpu','cuda_variant_assessed':False,
            'primary_plan':read('action-bound-consistency-v1-plan.json'),
            'extension_plan':read('action-bound-consistency-v1-extension-plan.json'),
            'completion':completion,'matched_comparison':comparison,'runs':runs,
            'all_thresholds_and_original_denominators_preserved':True,'full_goal_complete':False,
            'reference_bridge_diagnostic':read('evidence/continuous-reference-bridge-diagnostic.json'),
            'next_trial':read('execution.json',{}).get('next_controlled_trial'),
            'next_trial_plan':read('retention-trial-v1-plan.json'),
            'engineering_check':read('runs/engineering-retention-v2-cpu-6/proof.json')}
    (ROOT/'action-bound-ablation-status.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    payload=json.dumps(ledger,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同权重动作限幅对照</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:#142638;font:15px/1.65 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px}.card{background:white;border:1px solid #dbe4ec;border-radius:10px;padding:18px;margin:16px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dde4eb;vertical-align:top}th{background:#e8eff6;white-space:nowrap}.pass{color:#145539}.fail{color:#9b3425}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#0868a5}video{width:100%;max-width:760px}summary{cursor:pointer}small{color:#52677b}h1{line-height:1.3}</style>
<main><h1>同一组权重：输出限幅是否解决失败？</h1><p><a href="unified-library-report.html">返回三阶段完整报告</a> · <a href="action-bound-consistency-v1-plan.json">首轮冻结方案</a> · <a href="action-bound-consistency-v1-extension-plan.json">原始参考与连续序列扩展方案</a></p>
<div class="card"><strong>受控变量：</strong>只将采集阶段已有的 ±20 输出界限用于部署。权重、原始/校准参考、物理配置、奖励和门槛均保持。这里是独立控制器变体的 CPU 对照；没有训练更新，也没有该变体的 CUDA 评测。动作值是关节目标归一化前的策略输出，不是力矩或弧度。所有评测零重置，失败后保留完整尾段。</div>
<div id="status" class="card"></div><h2>完整分组与严格验收</h2><div id="groups" class="scroll"></div>
<h2>首次失败与超限的先后关系</h2><p>表格列出每一项首次失败。超限若在首次失败之后，不能解释该次初始失败。早于失败也不证明因果，需要看同权重干预的结果。状态差异阈值是 1e-5。</p><p>下图包括全部三条固定序列的首次失败窗口。完整120秒轨迹与视频另行保留；局部窗口用于定位失败。</p><img src="figures/continuous-first-failure-diagnostic.png" alt="全部三条连续序列：限幅前后的首次失败位置和姿态对照" style="width:100%;height:auto"><div id="timing" class="scroll"></div>
<h2>全部 55 条校准参考对照与原始新 15 条</h2><div id="clips"></div><h2>固定三条 120 秒序列：原始与校准参考</h2><div id="continuous"></div>
<h2>全部变体视频（含失败尾段）</h2><div id="videos"></div><h2>转场参考与下一步的边界</h2><div id="next" class="card"></div><details><summary>完整账本与观测计数</summary><pre id="details"></pre></details><p id="updated"></p>
<script>const D=PAYLOAD;const esc=x=>String(x??'—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');const num=x=>x==null?'—':Number(x).toFixed(3);const pass=x=>x==null?'待验收':'<span class="'+(x?'pass':'fail')+'">'+(x?'通过':'失败')+'</span>';
const find=mode=>D.runs.find(r=>r.mode===mode);
document.querySelector('#status').innerHTML='<strong>收尾状态：</strong>'+esc(D.completion?.status)+'。四组同后端对照：'+esc(D.matched_comparison?.status)+'。整体目标仍未完成。';
document.querySelector('#groups').innerHTML='<table><thead><tr><th>完整评测</th><th>新15原门槛</th><th>新15严格跟踪+接触</th><th>旧40原门槛</th><th>旧40均分</th><th>连续严格</th></tr></thead><tbody>'+D.runs.map(r=>'<tr><td>'+esc(r.mode)+'</td><td>'+(r.summary.groups?.new?.passed??'—')+(r.summary.groups?.new?'/15':'')+'</td><td>'+(r.quality?.combined_engineering_passed??'—')+(r.quality?.new_denominator?'/15':'')+'</td><td>'+(r.summary.groups?.lafan?.passed??'—')+(r.summary.groups?.lafan?'/40':'')+'</td><td>'+num(r.summary.groups?.lafan?.macro_return_original)+'</td><td>'+(r.quality?.continuous_engineering_passed??'—')+(r.summary.trials?'/3':'')+'</td></tr>').join('')+'</tbody></table>';
document.querySelector('#timing').innerHTML=(D.matched_comparison?.comparisons??[]).map(p=>'<details class="card" open><summary>'+esc(p.bounded)+'</summary><table><thead><tr><th>片段/序列</th><th>未限幅首次失败 s</th><th>首次输出超限 s</th><th>限幅后首次失败 s</th><th>失败前超限元素</th><th>仅在失败后超限</th></tr></thead><tbody>'+p.rows.filter(r=>!r.baseline_original_hard_gate_passed).map(r=>'<tr><td>'+esc(r.clip)+'</td><td>'+num(r.baseline_first_failure_s)+'</td><td>'+num(r.first_baseline_output_exceeds20_s)+'</td><td>'+num(r.bounded_first_failure_s)+'</td><td>'+r.exceeding_action_elements_before_first_failure+'</td><td>'+esc(r.overshoot_only_after_first_failure)+'</td></tr>').join('')+'</tbody></table></details>').join('');
for(const [baseline,bounded] of [['unified-stage3-corrected-cpu-all55','stage3-actionbound20-corrected-cpu-all55'],['stage3-raw-cpu-new15-control','stage3-actionbound20-raw-cpu-new15']]){const a=find(baseline),b=find(bounded);if(!a||!b)continue;const quality=Object.fromEntries((b.quality?.clips??[]).map(r=>[r.clip,r]));document.querySelector('#clips').innerHTML+='<details class="card"><summary>'+esc(bounded)+'：全部 '+b.summary.clips.length+' 条</summary><table><thead><tr><th>片段</th><th>组别</th><th>未限幅原门槛/均分</th><th>限幅原门槛/均分</th><th>限幅严格跟踪+接触</th></tr></thead><tbody>'+a.summary.clips.map((r,i)=>{const q=b.summary.clips[i];return '<tr><td>'+esc(r.clip)+'</td><td>'+esc(r.group)+'</td><td>'+pass(r.passed_original_hard_gate)+' / '+num(r.norm_return_original)+'</td><td>'+pass(q.passed_original_hard_gate)+' / '+num(q.norm_return_original)+'</td><td>'+pass(quality[r.clip]?.combined_engineering_passed)+'</td></tr>'}).join('')+'</tbody></table></details>'}
document.querySelector('#continuous').innerHTML=D.runs.filter(r=>r.summary.trials).map(r=>'<details class="card"><summary>'+esc(r.mode)+'：'+r.summary.passed_original_hard_gate+'/3 原门槛</summary><p>严格跟踪 '+(r.quality?.continuous_engineering_passed??'待验收')+'/3；补充几何接触 '+(r.contact?.contact_review_passed??'待验收')+'/3。每条120秒，零重置。未限幅第三阶段的连续接触审查为事后补充诊断；未测接触力。</p><pre>'+esc(JSON.stringify({trials:r.summary.trials,tracking:r.quality,contact:r.contact},null,2))+'</pre><a href="runs/'+esc(r.mode)+'/continuous-contact-review.json">全部20ms接触样本</a></details>').join('');
document.querySelector('#videos').innerHTML=D.runs.filter(r=>r.videos.length).map(r=>'<details class="card"><summary>'+esc(r.mode)+'：'+r.videos.length+' 段</summary>'+r.videos.map(v=>'<details><summary>'+esc(v.clip??v.file)+' · '+num(v.physical_window_seconds)+'s</summary><video controls preload="none" src="'+esc(v.file)+'"></video><br><a href="'+esc(v.file)+'">打开完整视频</a> · '+(v.complete_decode_passed?'全片解码通过':'待核验')+'</details>').join('')+'</details>').join('');
document.querySelector('#next').innerHTML='<p>三条连续序列的前两次初始失败发生在88_08、88_07动作内部；第三次发生在进入90_14的转场。全部固定转场均已检查：未发现关节目标超出模型关节范围，但存在参考穿地及采样无接触时的加速度疑点。这是几何/运动学诊断，尚不能证明动态不可实现；未测接触力。</p><p><a href="evidence/continuous-reference-bridge-diagnostic.json">全部转场证据</a> · <a href="retention-trial-v1-plan.json">下一次有限试验方案</a> · <a href="runs/engineering-retention-v2-cpu-6/proof.json">同实现6步真实更新、优化器和限幅读回验证</a> · <a href="evidence/retention-native-method-diff.json">相对原生方法的两处目标改动</a></p><p>下一候选使用一个101D学生策略；教师仅用于训练时约束旧能力和三条退化翻转片段，推理不加载教师。最多1000次新更新、1张H100、程序上限90分钟，不自动追加。位置/网络和全账号资源检查满足后才能提交。6步工程通过不等于动作学会。</p><pre>'+esc(JSON.stringify(D.next_trial,null,2))+'</pre>';
document.querySelector('#details').textContent=JSON.stringify(D,null,2);document.querySelector('#updated').textContent='更新时间（学院）：'+new Date(D.updated_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'});
</script></main></html>'''
    document=template.replace('PAYLOAD',payload)
    (ROOT/'action-bound-ablation-report.html').write_text(document,encoding='utf8')
    (ROOT/'evidence/action-bound-report-script.js').write_text(document.split('<script>',1)[1].split('</script>',1)[0],encoding='utf8')
    print(json.dumps({'report':'action-bound-ablation-report.html','runs':len(runs),'comparison_status':comparison['status'],'complete_video_receipts':sum(len(row['videos']) for row in runs)}),flush=True)


if __name__=='__main__':main()
