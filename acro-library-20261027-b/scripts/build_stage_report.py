"""Evidence report for the complete first stage and explicitly versioned follow-up."""
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(name,default=None):
    path=ROOT/name
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default

def by_clip(rows):return {row['clip']:row for row in rows}

def video(folder,name):return read(f'videos/{folder}/{name}.json')

def main():
    execution=read('execution.json');manifest=read('library-manifest.json');audit=read('library-audit.json')
    modes={'old16k':'old-16k-all55-v3','specialist18k':'parent-18k-all55-v3','stage1cuda':'unified4000-cuda-all55',
           'stage1cpu':'unified4000-cpu-all55','stage2corrected':'unified-stage2-corrected-cuda-all55','stage2raw':'unified-stage2-raw-cuda-new15',
           'stage2cpu':'unified-stage2-corrected-cpu-all55','stage3corrected':'unified-stage3-corrected-cuda-all55',
           'stage3raw':'unified-stage3-raw-cuda-new15','stage3cpu':'unified-stage3-corrected-cpu-all55'}
    summaries={key:read(f'runs/{name}/summary.json') for key,name in modes.items()}
    lookups={key:by_clip(value['clips']) if value else {} for key,value in summaries.items()}
    assessments={key:read(f'runs/{name}/tracking-assessment.json',{'clips':[]}) for key,name in modes.items()}
    tracking={key:by_clip(value['clips']) for key,value in assessments.items()}
    combined={key:read(f'runs/{name}/combined-assessment.json') for key,name in modes.items()}
    combined_lookup={key:by_clip(value['clips']) if value else {} for key,value in combined.items()}
    clearance=by_clip(read('evidence/raw-ground-clearance.json',{'clips':[]})['clips'])
    proof=read('runs/unified-h100-4000/proof.json');rows=[]
    for source,entry in zip(manifest['files'],audit['clips']):
        name=Path(source['path']).stem
        rows.append({'clip':name,'description':entry['semantics']['description'],'primary_source':entry['semantics']['source'],
                     'duration_s':entry['duration_s'],'source':source,'flags':entry['flags'],'ground':clearance.get(name),
                     'previously_trained':source['training_status_before_phase']=='trained_single_clip_specialist',
                     'models':{key:lookup.get(name) for key,lookup in lookups.items()},
                     'tracking':{key:lookup.get(name) for key,lookup in tracking.items()},
                     'combined':{key:lookup.get(name) for key,lookup in combined_lookup.items()},
                     'reset_samples_stage1':proof['sampling']['clip_reset_samples'][name] if proof else None,
                     'reset_samples_this_phase':proof['sampling']['clip_reset_samples'][name] if proof else None,
                     'unified_training':'received_updates' if proof else 'pending_or_in_progress',
                     'sha256':source['sha256'],'source_url':source['source_url'],'source_path':source['path'],
                     'old16k':tracking['old16k'].get(name) or lookups['old16k'].get(name),
                     'acro18k':tracking['specialist18k'].get(name) or lookups['specialist18k'].get(name),
                     'unified':tracking['stage1cuda'].get(name),'unified_combined':combined_lookup['stage1cuda'].get(name),
                     'quality_status':('passed' if combined_lookup['stage1cuda'][name]['combined_engineering_passed'] else 'failed') if name in combined_lookup['stage1cuda'] else 'not_evaluated',
                     'video':video('unified4000-cuda-all55',name),
                     'stage2_video':video('unified-stage2-corrected-cuda-all55',name),
                     'stage2_raw_video':video('unified-stage2-raw-cuda-new15',name),
                     'stage3_video':video('unified-stage3-corrected-cuda-all55',name),
                     'stage3_raw_video':video('unified-stage3-raw-cuda-new15',name)})
    old=[{'clip':name,'models':{key:value.get(name) for key,value in lookups.items()}} for name,row in lookups['old16k'].items() if row['group']=='lafan']
    for row in old:
        row['failure_videos']={key:video(modes[key],row['clip']) for key in ['stage2corrected','stage2cpu','stage3corrected','stage3cpu']}
    continuous=[]
    for name in ['parent18k-frozen-continuous-v1','old16k-frozen-continuous-v1','unified4000-frozen-continuous-v1','unified4000-corrected-continuous-v1',
                 'stage2-frozen-continuous-v1','stage2-corrected-continuous-v1','stage3-frozen-continuous-v1','stage3-corrected-continuous-v1']:
        result=read(f'runs/{name}/summary.json')
        if result:
            result['video_receipts']={str(row['seed']):video(name,f'seed-{row["seed"]}') for row in result['trials']}
            result['engineering_assessment']=read(f'runs/{name}/continuous-assessment.json');continuous.append(result)
    ledger={'updated_at':datetime.now(timezone.utc).isoformat(),'denominator':15,'initially_untrained':14,'one_policy_target':True,
            'source_manifest_status_fields_are_pretraining_snapshot':True,
            'execution':execution,'summaries':summaries,'combined_assessments':combined,'clips':rows,'old_clips':old,'continuous_results':continuous,
            'raw_continuous_index':read('continuous-frozen-v1/index.json'),'corrected_continuous_index':read('continuous-corrected-v1/index.json'),
            'resource_verification_stage1':read('evidence/final-resource-verification-4000.json'),'transfer_verification':read('evidence/local-transfer-verification-4000.json'),
            'resource_verification_stage2':read('evidence/final-resource-verification-stage2.json'),
            'resource_verification_stage3':read('evidence/final-resource-verification-stage3.json'),
            'stage3_plan':read('stage3-plan.json'),'stage3_proof':read('runs/unified-h100-stage3-4000/proof.json') or read('evidence/stage3-update-proof-from-console.json'),
            'stage3_engineering_proof':read('runs/engineering-stage3-cpu-6/proof.json')}
    (ROOT/'library-status.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    payload=json.dumps(ledger,ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G1 统一动作库：完整训练与物理验收</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:#142638;font:15px/1.65 system-ui,sans-serif}main{max-width:1600px;padding:30px;margin:auto}h1{font-size:30px;line-height:1.3}h2{font-size:22px;margin-top:28px}.muted{color:#52677b}.card{background:#fff;border:1px solid #dbe4ec;border-radius:12px;padding:20px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.metric{font-size:30px;font-weight:700}.badge{display:inline-block;border-radius:5px;padding:1px 7px;background:#e8eef4;white-space:nowrap}.pass{background:#ddf2e6;color:#145539}.fail{background:#ffe5df;color:#8a301f}.pending{background:#fff0c6;color:#735114}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4ec;vertical-align:top}th{background:#e8eff6;white-space:nowrap}code,pre{font-size:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap}input,select,button{padding:8px;font:inherit;border:1px solid #b8c8d6;border-radius:6px;margin:0 6px 8px 0;background:white}a{color:#0868a5}video{width:100%;max-width:760px;background:#142438}details{margin:8px 0}summary{cursor:pointer}footer{margin:28px 0;color:#52677b}@media(max-width:850px){main{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:25px}}
</style><main><h1>一个策略：15 条新素材 + 40 条原有动作</h1><p class="muted">剩余 14 条动作的统一训练、旧能力回归、完整失败记录与连续动作测试。BONES 路线暂停；本页不代表论文规模复现或原视频效果已全部完成。</p>
<div class="card" id="status"></div><div class="grid" id="metrics"></div>
<div class="card"><strong>如何读结果</strong><p>原门限检查原实现的高度与姿态失败条件；跟踪质量另查全局位置、全身姿态、朝向、倒置覆盖及末段直立；接触审查另查非预期头/躯干/骨盆触地与穿地。合并通过要求跟踪质量和接触同时通过。零重置不等于零失败。</p><p>单动作质量与接触门限在第一阶段候选验收前已固定。所有失败均保留完整轨迹。CPU 与 CUDA/BF16 分别比较，不能把两个后端的差异当成文件丢失。</p><a href="acceptance-contract.json">跟踪标准</a> · <a href="contact-acceptance-contract.json">接触标准</a> · <a href="evidence/local-transfer-verification-4000.json">720 MB 归档及逐文件校验</a></div>
<h2>第一阶段真实训练日志</h2><div class="card"><img src="figures/stage1-training-curves.png" alt="第一阶段4000次更新的完整80条训练日志曲线" style="width:100%;height:auto"><p>完整80条原日志；灰色区间只更新世界模型。训练回报包含XY奖励变体，不能代替上方物理验收结果。<a href="figures/stage1-training-curves.svg">SVG</a> · <a href="figures/stage1-training-curves.pdf">PDF</a> · <a href="runs/unified-h100-4000/train_log_with_header.csv">原始数值</a></p></div><h2>第二阶段完整训练日志</h2><div class="card"><img src="figures/stage2-training-curves.png" alt="第二阶段完整4000更新训练日志" style="width:100%;height:auto"><p>完整80条原始日志，400次世界模型预热；训练指标与独立物理验收分开。<a href="figures/stage2-training-curves.svg">SVG</a> · <a href="figures/stage2-training-curves.pdf">PDF</a> · <a href="runs/unified-h100-stage2-4000/train_log_with_header.csv">原始数值</a></p></div><h2>完整15条分数对照</h2><div class="card"><img src="figures/new-library-stage-comparison.png" alt="完整15条动作原始奖励分数及原始门限失败标记" style="width:100%;max-width:1100px;height:auto"><p>保留全部15条；虚线只是固定分数门槛，超过它仍须通过跟踪、接触及连续测试。每次评测只有一个固定seed，未估计统计不确定性。<a href="figures/new-library-stage-comparison.svg">SVG</a> · <a href="figures/new-library-stage-comparison.csv">完整数值</a></p></div><h2>完整新库 15 条</h2><p>固定素材版本 <code>c493a897f2ca81153e980a65ee6f0faf7cb07487</code>。README 写 16 条，实际只有 15 个 CSV；其中 88_08 曾专项训练，另外 14 条此前未训练。每条都提供完整视频和失败项。素材清单中的状态是训练前冻结快照；当前结果以本页逐项指标与 library-status.json 为准。</p>
<p><input id="search" placeholder="搜索动作编号、描述"><select id="filter"><option value="all">全部15条</option><option value="untrained">此前未训练14条</option><option value="pass">第一阶段合并通过</option><option value="fail">第一阶段合并未通过</option></select><button id="export">导出完整 JSON</button><span id="count"></span></p>
<div class="card scroll"><table><thead><tr><th>动作 / 来源</th><th>时长 / 采样</th><th>原16K CPU</th><th>专项18K CPU</th><th>第一阶段 CUDA<br>原门限 / 分数</th><th>第一阶段严格验收<br>跟踪 / 接触 / 合并</th><th>第二阶段<br>校准参考 / 原始参考</th><th>第三阶段<br>校准 CUDA / CPU / 原始</th><th>完整视频与诊断</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>原有 40 条完整回归</h2><div class="card" id="old-summary"></div><details class="card"><summary>展开全部40条逐项对照</summary><div class="scroll"><table><thead><tr><th>动作</th><th>原16K CPU</th><th>专项18K CPU</th><th>第一阶段 CPU</th><th>第一阶段 CUDA</th><th>第二阶段 CUDA</th><th>第二阶段 CPU</th><th>第三阶段 CUDA</th><th>第三阶段 CPU</th><th>完整失败视频</th></tr></thead><tbody id="old-rows"></tbody></table></div></details>
<h2>三条 120 秒连续物理测试</h2><div class="card"><p>原冻结序列合计覆盖全部15条新素材，完整保留动作时长；只允许 t=0 初始化，转场只变参考，不放置机器人。完整120秒执行后再统计，保留全部失败尾段。第一阶段原始参考结果为1/3通过原门限；这不等于全部动作质量通过。</p><p>另存的校准序列仅应用两条既定高度修正及对应转场高度插值，XY、朝向、关节、顺序和时间均不变。原始序列结果继续单独报告。训练未使用这两套冻结测试序列；转场动力学可行性尚未独立验证。</p><p>连续跟踪和末段速度标准在第一阶段结果之后定义、第二阶段候选评测前冻结；第一阶段对应计数仅作回顾性诊断。<a href="continuous-acceptance-contract-v1.json">连续标准及定义时点</a></p><a href="continuous-frozen-v1/index.json">原始冻结清单</a> · <a href="continuous-corrected-v1/index.json">校准对照清单</a></div><div id="continuous"></div>
<h2>第二阶段及解释边界</h2><div class="card" id="followup"></div>
<div class="card"><p>90_14、90_15 原始参考整段碰撞几何离地，地面校准分别平移约 -0.17949 m、-0.17120 m；不改原始CSV、动作节奏或关节。单独修正参考仍失败，修正本身不能算学会。89_01 只验身体姿态，没有平衡实物。接触检查是几何诊断，不是接触力、硬件稳定性或真实机器人安全验收。编译后的原生模型中手腕碰撞启用，橡胶手掌为视觉网格；不能将画面接地当作手掌支撑力已验证。<a href="evidence/runtime-hand-collision-inventory.json">碰撞模型核对</a>。</p><p>原16K基线文件仍保留，专项副本曾遗忘不代表原文件被覆盖。90/180/360类别与目标视频逐项覆盖尚未完成核验。</p><a href="execution.json">执行状态</a> · <a href="stage2-plan.json">固定续训方案</a> · <a href="library-manifest.json">素材及SHA256</a> · <a href="library-audit.json">全库审计</a> · <a href="target-coverage.json">目标视频覆盖</a> · <a href="evidence/final-resource-verification-4000.json">第一阶段资源释放核验</a> · <a href="evidence/unified4000-all15-visual-audit.jpg">15条画面抽查</a></div><h2>第三阶段：实际回放采样修正</h2><div class="card" id="stage3"></div><footer id="updated"></footer>
<script>const D=PAYLOAD;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=x=>x==null?'—':Number(x).toFixed(3);const mark=x=>`<span class="badge ${x==null?'pending':x?'pass':'fail'}">${x==null?'未验收':x?'通过':'失败'}</span>`;
const score=r=>r?mark(r.passed_original_hard_gate)+' / '+n(r.norm_return_original):mark(null);
const linkVideo=(v,label='完整物理视频')=>v?'<details><summary>'+esc(label)+'</summary><video controls preload="none" src="'+esc(v.file)+'"></video><br><a href="'+esc(v.file)+'">打开视频</a> · '+v.physical_window_seconds.toFixed(2)+'秒物理窗口；全片解码通过</details>':'视频尚未生成';
const counts=D.combined_assessments.stage1cuda;
document.querySelector('#status').innerHTML='<strong>当前状态：</strong>'+esc(D.execution.stage)+'<p>第一阶段已完成并取回全部44个归档清单文件。第二阶段：'+esc(D.execution.next_stage?.status??'未提交')+'；任务 <code>'+esc(D.execution.next_stage?.job_name??'—')+'</code>。这是保存时的快照，运行中不代表已学会。</p>';
const group=(k,g)=>D.summaries[k]?.groups[g];const c=group('stage1cpu','lafan'),g=group('stage1cuda','lafan');
document.querySelector('#metrics').innerHTML=[['新库原门限',counts?counts.original_hard_gate_passed+' / 15':'未验收','第一阶段 CUDA，完整失败尾段保留'],['新库严格合并',counts?counts.combined_engineering_passed+' / 15':'未验收','跟踪 '+(counts?.tracking_quality_passed??'—')+'，接触 '+(counts?.contact_review_passed??'—')],['旧库完整回归',c?c.passed+' / 40':'未验收','CPU均分 '+n(c?.macro_return_original)+'；CUDA '+(g?.passed??'—')+'/40'],['原始连续测试','1 / 3','每条120秒，均零重置'],['新增训练更新','4000','WM 800次预热后联合更新；55条均有采样'],['第二阶段',(D.execution.next_stage?.new_updates_observed??0)+' / 4000','已保存更新；校准参考；最终物理评测另列']].map(x=>'<div class="card"><small>'+x[0]+'</small><div class="metric">'+x[1]+'</div>'+x[2]+'</div>').join('');
function render(){const q=document.querySelector('#search').value.toLowerCase(),f=document.querySelector('#filter').value;const rows=D.clips.filter(r=>(r.clip+' '+r.description).toLowerCase().includes(q)&&(f==='all'||f==='untrained'&&!r.previously_trained||f==='pass'&&r.combined.stage1cuda?.combined_engineering_passed||f==='fail'&&r.combined.stage1cuda&&!r.combined.stage1cuda.combined_engineering_passed));document.querySelector('#count').textContent=`显示 ${rows.length} / 15，分母固定`;document.querySelector('#rows').innerHTML=rows.map(r=>{const a=r.combined.stage1cuda;return '<tr><td><a href="'+esc(r.source.source_url)+'">'+esc(r.clip.replace('_stageii_unitree_g1',''))+'</a><br>'+esc(r.description)+'<br><small>'+(r.previously_trained?'曾专项训练':'此前未训练')+' · <a href="'+esc(r.primary_source)+'">原始动作描述</a></small></td><td>'+r.duration_s.toFixed(2)+'s<br><small>训练重置起点采样 '+r.reset_samples_stage1+'</small></td><td>'+score(r.models.old16k)+'</td><td>'+score(r.models.specialist18k)+'</td><td>'+score(r.models.stage1cuda)+'</td><td>跟踪 '+mark(a?.tracking_quality_passed)+'<br>接触 '+mark(a?.contact_review_passed)+'<br>合并 '+mark(a?.combined_engineering_passed)+'</td><td>校准 '+score(r.models.stage2corrected)+'<br>合并 '+mark(r.combined.stage2corrected?.combined_engineering_passed)+'<br>原始 '+score(r.models.stage2raw)+'<br>合并 '+mark(r.combined.stage2raw?.combined_engineering_passed)+'</td><td>CUDA '+score(r.models.stage3corrected)+'<br>合并 '+mark(r.combined.stage3corrected?.combined_engineering_passed)+'<br>CPU '+score(r.models.stage3cpu)+'<br>合并 '+mark(r.combined.stage3cpu?.combined_engineering_passed)+'<br>原始 '+score(r.models.stage3raw)+'<br>合并 '+mark(r.combined.stage3raw?.combined_engineering_passed)+'</td><td>'+(r.stage3_video?linkVideo(r.stage3_video,'第三阶段校准参考视频'):'')+(r.stage3_raw_video?linkVideo(r.stage3_raw_video,'第三阶段原始参考视频'):'')+(r.stage2_video?linkVideo(r.stage2_video,'第二阶段校准参考视频'):'')+(r.stage2_raw_video?linkVideo(r.stage2_raw_video,'第二阶段原始参考视频'):'')+linkVideo(r.video,'第一阶段原始参考视频')+(r.ground?.all_frames_collision_geometry_above_ground_2cm?'<br><span class="badge fail">原参考整段离地</span>':'')+'<details><summary>指标、门限和失败项</summary><pre>'+esc(JSON.stringify({tracking:r.tracking.stage1cuda,contact_and_combined:a,stage2_corrected_tracking:r.tracking.stage2corrected,stage2_corrected_combined:r.combined.stage2corrected,stage2_raw_tracking:r.tracking.stage2raw,stage2_raw_combined:r.combined.stage2raw,stage2_cpu_combined:r.combined.stage2cpu,stage3_corrected_tracking:r.tracking.stage3corrected,stage3_corrected_combined:r.combined.stage3corrected,stage3_cpu_combined:r.combined.stage3cpu,stage3_raw_combined:r.combined.stage3raw,reference_flags:r.flags,source_sha256:r.source.sha256},null,2))+'</pre></details></td></tr>'}).join('')}
document.querySelector('#search').oninput=render;document.querySelector('#filter').onchange=render;render();
document.querySelector('#old-summary').innerHTML='<p>原16K CPU：40/40，均分0.845；专项18K CPU：2/40，均分0.173。第一阶段 CPU：'+(c?.passed??'—')+'/40，均分'+n(c?.macro_return_original)+'；CUDA/BF16：'+(g?.passed??'—')+'/40，均分'+n(g?.macro_return_original)+'。CUDA 的 jumps1_subject2 在212.96秒首次触发原门限，CPU同一条通过，后端差异保留。</p>';
document.querySelector('#old-rows').innerHTML=D.old_clips.map(r=>'<tr><td>'+esc(r.clip)+'</td>'+['old16k','specialist18k','stage1cpu','stage1cuda','stage2corrected','stage2cpu','stage3corrected','stage3cpu'].map(k=>'<td>'+score(r.models[k])+'</td>').join('')+'<td>'+Object.entries(r.failure_videos).filter(([k,v])=>v).map(([k,v])=>linkVideo(v,k)).join('')+'</td></tr>').join('');
document.querySelector('#continuous').innerHTML=D.continuous_results.map(run=>'<details class="card" '+(run.mode==='unified4000-frozen-continuous-v1'?'open':'')+'><summary>'+esc(run.mode)+'：'+run.passed_original_hard_gate+'/3 原门限通过</summary><p>参考版本 '+esc(run.reference_version)+'；完整360秒；重置 '+run.post_initialization_resets+'；'+esc(run.device)+'</p>'+run.trials.map(t=>'<div class="card"><strong>seed '+t.seed+'</strong> '+mark(t.passed_original_hard_gate)+'；首次失败 '+(t.first_hard_failure_s==null?'无':t.first_hard_failure_s.toFixed(2)+'s')+'；XY RMSE '+n(t.root_xy_rmse_m)+'m；峰值 '+n(t.root_xy_peak_m)+'m'+linkVideo(run.video_receipts[String(t.seed)])+'<details><summary>完整转场和末段指标</summary><pre>'+esc(JSON.stringify(t,null,2))+'</pre></details></div>').join('')+'</details>').join('');
document.querySelector('#followup').innerHTML='<p>第二阶段只对90_14、90_15使用独立版本的恒定高度校准。继续一个101D策略与完整WM/价值网络，旧归一化统计固定，60%新库与40%旧库为重置起点先验；不是保证的梯度占比。计划4000次新更新、400次WM预热、1024环境。校准55条与原始新库15条分别评测，不降低门限，不用更新数推断成功。</p><pre>'+esc(JSON.stringify(D.execution.next_stage,null,2))+'</pre>';

const stage2cpu=D.combined_assessments.stage2cpu,stage2cuda=D.combined_assessments.stage2corrected;
document.querySelector('#stage3').innerHTML='<p>第二阶段校准新15严格合并：CUDA '+(stage2cuda?.combined_engineering_passed??'待验收')+'/15；CPU '+(stage2cpu?.combined_engineering_passed??'待验收')+'/15。CPU/CUDA旧40均分别报告。</p><p>第三阶段修正实际WM窗口起点和有效策略起点的回放采样：新库60%，旧库40%；新库按片段均分，第二阶段CPU或CUDA失败的3条旧片段固定3倍优先级。重置先验、参考、架构、奖励、归一化张量和验收门槛不变。6次真实CPU更新与保存读回已验证。运行中不代表已通过物理验收。</p><p><a href="stage3-plan.json">冻结方案</a> · <a href="evidence/balanced-replay-tests.json">采样及边界检查</a> · <a href="runs/engineering-stage3-cpu-6/proof.json">真实更新与读回证明</a> · <a href="evidence/final-resource-verification-stage2.json">第二阶段资源释放</a></p><pre>'+esc(JSON.stringify(D.execution.stage3??{},null,2))+'</pre><h3>完整55条分组结果</h3><pre>'+esc(JSON.stringify({stage2_cpu:D.summaries.stage2cpu?.groups,stage2_cuda:D.summaries.stage2corrected?.groups,stage3_cpu:D.summaries.stage3cpu?.groups,stage3_cuda:D.summaries.stage3corrected?.groups,stage3_raw_new:D.summaries.stage3raw?.groups,stage3_corrected_combined:D.combined_assessments.stage3corrected,stage3_raw_combined:D.combined_assessments.stage3raw},null,2))+'</pre>';
document.querySelector('#updated').textContent='快照时间（学院）：'+new Date(D.updated_at).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'})+'。视频：灰色是真实物理轨迹，绿色参考仅为显示向Y平移1.5m，指标按未平移世界坐标计算。';
document.querySelector('#export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(D,null,2)],{type:'application/json'}));a.download='g1-library-status.json';a.click();URL.revokeObjectURL(a.href)};
</script></main></html>'''
    if (ROOT/'action-bound-ablation-report.html').exists():
        anchor='<div class="card" id="status"></div>'
        assert anchor in template
        latest='<section class="card"><h2>最新诊断：同权重动作限幅</h2><p>第三阶段校准 CUDA 严格合并 11/15，原始 CUDA 7/15；旧库 CUDA 38/40、CPU 37/40；原始与校准连续严格均 0/3，整体目标未完成。</p><a href="action-bound-ablation-report.html">查看独立控制器变体 CPU 完整对照、原始/校准新15、旧40、固定120秒序列与失败时序</a></section>'
        template=template.replace(anchor,latest+anchor,1)
    html=template.replace('PAYLOAD',payload)
    (ROOT/'unified-library-report.html').write_text(html,encoding='utf8')
    script=html.split('<script>',1)[1].split('</script>',1)[0];(ROOT/'evidence/report-script-syntax-check.js').write_text(script,encoding='utf8')
    print(json.dumps({'report':str(ROOT/'unified-library-report.html'),'new_rows':len(rows),'old_rows':len(old),'stage1_video_receipts':sum(row['video'] is not None for row in rows),
                      'continuous_runs':len(continuous),'raw_new_combined_passes':combined['stage1cuda']['combined_engineering_passed'] if combined['stage1cuda'] else None}),flush=True)

if __name__=='__main__':main()
