"""Report the complete bounded-teacher diagnostic and its fixed denominators."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf8'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    execution=read('evidence/teacher-qualification-1030-execution.json')
    assert execution['status']=='complete_all30_full_tail_videos_decoded_hashed'
    assert execution['video_count']==30 and execution['wall_seconds']<=1800
    assert execution['weights_and_all_frozen_contract_source_hashes_unchanged']
    plan=read('evidence/retention-1030-next-diagnostic-plan.json')
    runs=[]
    for variant,mode in execution['modes'].items():
        runs.append({'variant':variant,'mode':mode,'summary':read(f'runs/{mode}/summary.json'),
                     'tracking':read(f'runs/{mode}/tracking-assessment.json'),
                     'quality':read(f'runs/{mode}/combined-assessment.json'),
                     'contact':read(f'runs/{mode}/contact-review.json'),
                     'videos':[value for value in execution['videos'] if value['source_run']==mode]})
    protected=[]
    raw,corrected=runs
    with np.load(ROOT/f'runs/{raw["mode"]}/trajectory.npz') as first,np.load(ROOT/f'runs/{corrected["mode"]}/trajectory.npz') as second:
        for name in plan['protected_new']:
            index=next(i for i,row in enumerate(raw['summary']['clips']) if row['clip']==name)
            stop=raw['summary']['clips'][index]['full_clip_steps']+1
            same={key:bool(np.array_equal(first[key][:stop,index],second[key][:stop,index],equal_nan=True))
                  for key in ['state','reference_qpos','action','xy_error']}
            assert all(same.values()),'Protected raw/calibrated references are unchanged and matched trajectories must agree'
            entry={'clip':name,'raw_calibrated_trajectory_identity':same,'outcomes':{}}
            for run in runs:
                physical=next(row for row in run['summary']['clips'] if row['clip']==name)
                quality=next(row for row in run['quality']['clips'] if row['clip']==name)
                tracking=next(row for row in run['tracking']['clips'] if row['clip']==name)
                contact=next(row for row in run['contact']['clips'] if row['clip']==name)
                bound=next(row for row in run['summary']['action_bound_observations'] if row['clip']==name)
                entry['outcomes'][run['variant']]={'physical':physical,'quality':quality,'tracking':tracking,'contact':contact,'action_bound':bound}
            protected.append(entry)
    comparisons=[]
    for label,modes in [('Stage2 teacher',execution['modes']),('Frozen stage3 parent',{'raw':'stage3-actionbound20-raw-cpu-new15','corrected':'stage3-actionbound20-corrected-cpu-all55'}),
                        ('Unadopted1000-update candidate',{'raw':'retention-v1-raw-cpu-new15','corrected':'retention-v1-corrected-cpu-all55'})]:
        for variant,mode in modes.items():
            q=read(f'runs/{mode}/combined-assessment.json')
            comparisons.append({'model':label,'reference':variant,'mode':mode,'strict_passed':q['combined_engineering_passed'],
                                'denominator':15,'protected_strict_passed':sum(row['combined_engineering_passed'] for row in q['clips'] if row['clip'] in plan['protected_new']),
                                'protected_denominator':3,'source_sha256':sha(ROOT/f'runs/{mode}/combined-assessment.json')})
    result={'status':'complete_fixed_stage2_teacher_not_qualified_for_protected_preservation' if not execution['protected_teacher_qualified'] else 'complete_protected_teacher_qualified',
            'updated_at':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':execution['checkpoint_sha256'],'plan_sha256':execution['plan_sha256'],
            'GPU':0,'training_updates':0,'CPU_affinity':execution['CPU_affinity'],'program_wall_seconds':execution['wall_seconds'],
            'new_denominator_per_reference':15,'protected_denominator_per_reference':3,'groups':execution['groups'],
            'protected':protected,'comparisons':comparisons,'videos_complete':30,'all_failed_tails_retained':True,
            'all_post_t0_resets':0,'protected_teacher_qualified':execution['protected_teacher_qualified'],
            'candidate_adopted':False,'full_goal_complete':False,'CUDA_teacher_diagnostic_not_executed':True,
            'inferences':['The fixed stage2 teacher cannot be treated as a qualified closed-loop target for all three protected motions under these unchanged strict contracts.',
                          'Lower action-imitation error alone does not establish physical acceptance; this diagnostic does not identify the cause of the previous student regression.',
                          'Raw/calibrated protected references, actions and complete trajectories are identical; the existing height calibration does not repair these protected failures.',
                          'No stronger teacher loss, second identical1000-update block, new GPU job or altered reference/physics/threshold follows automatically.']}
    target=ROOT/'evidence/teacher-qualification-1030-assessment.json'
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors=['#325a86','#9aaec1','#a6532e']
    for axis,variant in zip(axes,['raw','corrected']):
        values=[row for row in comparisons if row['reference']==variant]
        bars=axis.bar(range(3),[row['strict_passed'] for row in values],color=colors,width=.6)
        axis.bar_label(bars,labels=[str(row['strict_passed'])+'/15' for row in values],padding=5)
        axis.set_ylim(0,16);axis.set_xticks(range(3),['Stage2 teacher','Stage3 parent','Unadopted candidate']);axis.tick_params(axis='x',labelsize=9)
        axis.set_ylabel('Complete new clips passing tracking and contact');axis.set_title('CPU / '+('raw' if variant=='raw' else 'calibrated'))
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
        axis.text(.5,.91,'Protected motions: 0/3 for every model',transform=axis.transAxes,ha='center',fontsize=9,color='#9b3e27')
    fig.suptitle('Stage2 teacher qualification: same bound20, fixed15-clip denominator',fontsize=13)
    artifacts=[]
    for suffix in ['png','svg','pdf']:
        p=ROOT/f'figures/teacher-qualification-1030.{suffix}';fig.savefig(p,dpi=170)
        artifacts.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    plt.close(fig)
    (ROOT/'evidence/teacher-qualification-1030-plot-verification.json').write_text(json.dumps({'plots':artifacts,'source_sha256':sha(target),'fixed_denominators':True},indent=2)+'\n')
    reward_audit=read('evidence/protected-reward-audit-1030.json')
    geometry=read('evidence/reference-geometry-1030.json')
    specialist=read('evidence/specialist-teacher-8808-1030.json')
    assert specialist['status']=='complete_two_full_clip_specialist_controls'
    engineering=read('runs/engineering-qualified8808-cpu-6-1030/proof.json')
    readback=read('evidence/qualified-expert-engineering-1030-readback.json')
    assert engineering['new_updates']==6 and readback['exact_output_equality']
    payload={'assessment':result,'runs':runs,'reward_audit':reward_audit,'reference_geometry':geometry,'specialist_control':specialist,
             'engineering':engineering,'engineering_readback':readback,'next_proposal':read('evidence/qualified-expert-retention-v2-proposal.json'),
             'model_download':'https://github.com/skywhite1024/SuperMimic-Reproduction/releases/download/v0.1.0/unified-stage2-full.pt'}
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SuperMimic · 教师资格诊断</title><style>
    :root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#14263b;background:#f3f6fa}body{margin:0}main{max-width:1120px;margin:auto;padding:32px 22px}h1{font-size:30px}h2{margin-top:30px}.card,details{background:white;border:1px solid #dce3eb;border-radius:10px;padding:18px;margin:14px 0}.bad{color:#a33d27;font-weight:650}.good{color:#21694c}.muted{color:#5c6d7f}table{width:100%;border-collapse:collapse;background:white}th,td{padding:11px;border-bottom:1px solid #dce3eb;text-align:left}th{background:#eaf0f6}select,input,button{padding:9px;margin:4px;border:1px solid #abb8c6;border-radius:6px;background:white;color:inherit}video,img{max-width:100%;height:auto}video{display:block;margin-top:12px;width:960px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:520px;overflow:auto}a{color:#255d91}.scroll{overflow:auto}.row{display:flex;gap:12px;flex-wrap:wrap}.stat{flex:1;min-width:180px;background:white;padding:18px;border:1px solid #dce3eb;border-radius:10px}.stat b{font-size:28px;display:block}
    </style><main><a href="retention-1030-report.html">← 1000更新候选完整记录</a><h1>相同部署限幅下，stage2教师是否合格？</h1>
    <p>本机CPU，0次训练更新，固定101D权重与±20输出限幅。原始15条和既定高度校准15条均从frame0原生初始化，t0后不重置，失败尾段完整保留。</p>
    <div class="card bad" id="conclusion"></div><div class="row" id="stats"></div><h2>完整分母与三条保护动作</h2><div class="scroll" id="counts"></div>
    <p class="muted">原始门槛、严格跟踪、几何接触分别验收；接触力未测量。所有结果属于社区实现的工程诊断，整体55动作和连续120秒目标仍未完成。</p>
    <img src="figures/teacher-qualification-1030.png" alt="同一CPU后端、同一15动作分母下教师、父模型和未采用候选的对比图">
    <h2>全部30条完整视频和逐项指标</h2><select id="variant" aria-label="选择参考"><option value="raw">原始参考 ·15条</option><option value="corrected">高度校准 ·15条</option></select><input id="search" aria-label="搜索动作" placeholder="搜索动作"><select id="filter" aria-label="筛选结果"><option value="all">全部15条</option><option value="protected">三条保护动作</option><option value="failed">严格未通过</option></select><span id="count"></span><div class="scroll" id="clips"></div>
    <h2>奖励与参考几何的只读审查</h2><div class="card" id="reward-audit"></div><div class="card" id="specialist"></div><div class="card" id="engineering"></div>
    <h2>可复核证据</h2><div class="card"><a href="evidence/teacher-qualification-1030-execution.json">完整有限程序收据及30份视频哈希</a> · <a href="evidence/teacher-qualification-1030-assessment.json">全部结论与指标</a> · <a href="evidence/retention-1030-next-diagnostic-plan.json">冻结的诊断计划</a><p id="model"></p><p>运行使用已有依赖与已核验输入；本次没有重新验证依赖安装或数据网络下载。历史未限幅结果完整保留，未替换原有学生评测。</p><details><summary>完整诊断账本</summary><button id="download">导出完整JSON</button><pre id="ledger"></pre></details></div>
    <script>const D=PAYLOAD;
    const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const mark=x=>'<span class="'+(x?'good':'bad')+'">'+(x?'通过':'未通过')+'</span>';
    document.querySelector('#conclusion').textContent=D.assessment.protected_teacher_qualified?'三条保护动作均已通过匹配验收；整体目标另行验收。':'教师在两套参考下均为保护动作0/3，不能作为三条动作的合格保留目标。原1000更新候选仍不采用。';
    document.querySelector('#stats').innerHTML=[['原始严格',D.assessment.groups.raw.combined_engineering_passed+'/15'],['校准严格',D.assessment.groups.corrected.combined_engineering_passed+'/15'],['完整视频','30/30'],['新增训练更新','0']].map(([a,b])=>'<div class="stat">'+a+'<b>'+b+'</b></div>').join('');
    document.querySelector('#counts').innerHTML='<table><tr><th>参考</th><th>原始门槛</th><th>严格跟踪</th><th>几何接触</th><th>跟踪与接触</th><th>三条保护动作</th></tr>'+Object.entries(D.assessment.groups).map(([key,q])=>'<tr><td>'+esc(key)+'</td>'+['original_hard_gate_passed','tracking_quality_passed','contact_review_passed','combined_engineering_passed'].map(k=>'<td>'+q[k]+'/15</td>').join('')+'<td>'+q.protected_strict_passed+'/3</td></tr>').join('')+'</table>';
    function render(){const r=D.runs.find(x=>x.variant===document.querySelector('#variant').value);const term=document.querySelector('#search').value.toLowerCase(),f=document.querySelector('#filter').value;const rows=r.quality.clips.filter(x=>x.clip.toLowerCase().includes(term)&&(f!=='protected'||D.assessment.protected.some(y=>y.clip===x.clip))&&(f!=='failed'||!x.combined_engineering_passed));document.querySelector('#count').textContent='显示 '+rows.length+' /15；原始分母保持';document.querySelector('#clips').innerHTML='<table><tr><th>动作</th><th>原始门槛</th><th>跟踪</th><th>接触</th><th>完整证据</th></tr>'+rows.map(q=>{const v=r.videos.find(v=>v.clip===q.clip),p=r.summary.clips.find(p=>p.clip===q.clip),t=r.tracking.clips.find(t=>t.clip===q.clip);return '<tr><td>'+esc(q.clip)+'</td><td>'+mark(q.original_hard_gate_passed)+'</td><td>'+mark(q.tracking_quality_passed)+'</td><td>'+mark(q.contact_review_passed)+'</td><td><details><summary>完整视频 ·'+v.physical_window_seconds.toFixed(2)+'秒</summary><button class="video-toggle" type="button">播放 / 暂停</button><video controls preload="none" src="'+esc(v.file)+'"></video><a href="'+esc(v.file)+'">打开视频</a></details><details><summary>指标与失败原因</summary><pre>'+esc(JSON.stringify({physical:p,tracking:t,combined:q,video:v},null,2))+'</pre></details></td></tr>'}).join('')+'</table>'}
    ['variant','filter'].forEach(id=>document.querySelector('#'+id).onchange=render);document.querySelector('#search').oninput=render;render();
    document.addEventListener('click',e=>{const b=e.target.closest('.video-toggle');if(!b)return;const v=b.nextElementSibling;if(v.paused)v.play().catch(()=>b.textContent='无法播放，请使用视频链接');else v.pause()});
    document.querySelector('#model').innerHTML='固定教师 <a href="'+D.model_download+'">stage2完整模型</a>，SHA256：<code>'+D.assessment.checkpoint_sha256+'</code>。';
    const auditRows=D.reward_audit.rows.filter(x=>x.mode==='teacher-stage2-bound20-corrected-cpu-new15-1030');
    document.querySelector('#reward-audit').innerHTML='<p>18组既有保护动作记录的原生奖励逐项重算，最大奖励和误差 '+Math.max(...D.reward_audit.rows.map(x=>x.reward_reconstruction_abs_difference)).toExponential(2)+'；没有新物理轨迹或训练更新。下表保持原奖励及0.65门槛，不能把删除惩罚后的数值当作新成绩。</p><div class="scroll"><table><tr><th>教师动作</th><th>跟踪项</th><th>动作变化惩罚</th><th>关节限位惩罚</th><th>原始总分</th><th>参考最深穿透</th></tr>'+auditRows.map(x=>'<tr><td>'+esc(x.clip)+'</td><td>'+x.normalized_native_components.native_tracking.toFixed(3)+'</td><td>'+x.normalized_native_components.action_rate_penalty.toFixed(3)+'</td><td>'+x.normalized_native_components.joint_limit_penalty.toFixed(3)+'</td><td>'+x.original_normalized_return.toFixed(3)+'</td><td>'+(-100*x.reference_ground_distance_min_after_100ms_m).toFixed(2)+'cm</td></tr>').join('')+'</table></div><p>三条保护动作的参考均未超过硬关节限位；88_08与88_09在参考姿态中出现超过5cm的几何地面穿透。几何检查不证明动态可行或不可行，也不允许据此修改参考或降低验收门槛。</p><a href="evidence/protected-reward-audit-1030.json">完整奖励审查</a> · <a href="evidence/reference-geometry-1030.json">全部30份参考几何记录</a>';
    document.querySelector('#specialist').innerHTML='<h3>旧88_08单动作模型的匹配控制</h3><p>这是99D单动作模型，与101D统一教师的架构和训练范围不同；每套参考分母仅1，不能合并为15/55动作成绩或替换统一学生。</p>'+Object.entries(D.specialist_control.groups).map(([variant,g])=>'<p>'+esc(variant)+'：严格 '+mark(g.assessment.combined_engineering_passed)+'，原始总分 '+g.assessment.norm_return_original.toFixed(3)+'，最深实际穿透 '+(-100*g.assessment.actual_min_after_initial_100ms_m).toFixed(2)+'cm。</p><details><summary>单动作模型完整视频 ·'+g.video.physical_window_seconds.toFixed(2)+'秒</summary><button class="video-toggle" type="button">播放 / 暂停</button><video controls preload="none" src="'+esc(g.video.file)+'"></video></details>').join('')+'<a href="evidence/specialist-teacher-8808-1030.json">两条完整控制记录与全部失败原因</a>';
    document.querySelector('#engineering').innerHTML='<h3>另行隔离的6次CPU工程更新</h3><p>教师资格诊断本身0次训练更新。此后仅在独立目录完成2次世界模型更新和4次联合更新，验证选择性88_08教师接口；没有替换父模型或原1000更新候选。世界模型、策略和价值网络的优化器增量分别为6、4、4，教师与所有归一化张量不变，4份真实历史经独立加载器的限幅输出逐位一致。工程通过不证明物理表现改善。</p><p>原读回探针的首版因4份历史与1份计数形状不匹配而失败；修正探针形状后通过，失败记录保留。另一个零更新接口探针使用单动作模型的真实状态，历史字段名的更正另附收据。</p><a href="runs/engineering-qualified8808-cpu-6-1030/proof.json">6次实际更新证明</a> · <a href="evidence/qualified-expert-engineering-1030-readback.json">独立加载器读回</a> · <a href="evidence/qualified-teacher-adapter-1030-clarification.json">接口探针字段说明</a><p>下一项研究提案：仅对已合格88_08使用该教师，88_07/09继续原生目标；保持单个101D学生、旧库约束及完整分母。最多1张H100、1000次新更新、5400秒；当前未提交，仍须准备并核对有限运行命令及CUDA教师先验收。</p><a href="evidence/qualified-expert-retention-v2-proposal.json">具体范围、预算与停止条件</a>';
    const ledger=JSON.stringify(D,null,2);document.querySelector('#ledger').textContent=ledger.length>60000?'界面仅预览前60000字符；导出保留全部记录。\\n'+ledger.slice(0,60000):ledger;
    document.querySelector('#download').onclick=()=>{const u=URL.createObjectURL(new Blob([ledger],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='teacher-qualification-1030-ledger.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
    </script></main></html>'''
    html=template.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    (ROOT/'teacher-qualification-1030-report.html').write_text(html,encoding='utf8')
    (ROOT/'teacher-qualification-1030-status.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'protected_strict':{key:value['protected_strict_passed'] for key,value in result['groups'].items()},'videos':30},indent=2))

if __name__=='__main__':
    main()
