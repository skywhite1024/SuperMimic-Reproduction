"""Standalone local experiment report; plots are backed by retained numeric trajectories."""
import html
import json
import re
from pathlib import Path
import numpy as np
from runtime import ROOT


def main():
    modes={}
    for p in sorted((ROOT/'runs').glob('*/summary.json')):
        if not (p.parent/'trajectory.npz').is_file():continue
        summary=json.loads(p.read_text())
        if 'trials' not in summary:continue
        with np.load(p.parent/'trajectory.npz') as a:
            rows=[]
            for i,t in enumerate(summary['trials']):
                rows.append({'metrics':t,'xy':a['xy_error'][::10,i].tolist(),'local':a['body_local'][::10,i].tolist(),'global':a['body_global'][::10,i].tolist(),'yaw':np.rad2deg(a['yaw_error'][::10,i]).tolist(),'fail':a['hard_failure'][::10,i].astype(int).tolist()})
        modes[p.parent.name]={'summary':summary,'series':rows}
    manifest_set='references-lookahead' if 'global-adapter-600-lookahead' in modes else 'references'
    manifest=json.loads((ROOT/manifest_set/'seed-0/manifest.json').read_text())
    acro=json.loads((ROOT/'runs/acro-baseline/summary.json').read_text())
    data=json.dumps({'modes':modes,'segments':manifest['segments'],'acro':acro}).replace('</','<\\/')
    target=json.loads((ROOT/'evidence/target-video.json').read_text(encoding='utf-8'))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 连续多动作复现</title>
<style>:root{color-scheme:light}body{margin:0;background:#f2f5f9;color:#182d43;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:auto;padding:30px 24px 80px}h1{font-size:32px;line-height:1.25}h2{margin-top:35px;font-size:23px}p{max-width:1050px}.status{background:#fff1c9;border-left:5px solid #bb7e16;padding:16px 20px;border-radius:7px}video{display:block;width:100%;background:#152433;border-radius:10px;margin-top:22px}.grid{display:flex;gap:15px;flex-wrap:wrap}.card{flex:1;min-width:180px;padding:17px;background:white;border-radius:9px}.card b{display:block;font-size:27px}.meta{font-size:14px;color:#50677d}select,button{font:inherit;border:1px solid #c6d3df;border-radius:6px;background:white;padding:7px 12px;color:#193f5e}button{cursor:pointer}button:hover{background:#e8f2fa}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4ec}th{background:#e7eef5}svg{display:block;width:100%;background:white;border-radius:9px;margin:12px 0}#segments{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:7px}#segments button{font-size:13px;text-align:left}a{color:#126b9e}code{overflow-wrap:anywhere}details{padding:12px 18px;background:white;border-radius:9px;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.bad{color:#a23331}.good{color:#176c43}</style>
<main><p class="meta">2026-10-27 · 独立实验 interactive-g1-20261027-a</p><h1>G1 连续多动作闭环复现</h1>
<div class="status"><b>已得到基础动作版；全部目标效果尚未完成。</b><br>原 16000 检查点在 3 条预先固定的 120 秒 / 29 段序列上均为初始化后零重置、零原硬门限失败。全局漂移仍明显，空翻单例未通过。位置增强试训的效果须另行评测，不能用训练更新次数代替成功证据。</div>
<video id="video" controls preload="metadata" src="videos/baseline-seed0.mp4"></video><p class="meta">视频固定为 baseline / seed 0，来自同一条连续物理轨迹。绿色参考仅有固定 +1.5m Y 显示偏移；全部误差使用未偏移的固定世界坐标。下方选择器改变数值图表，不切换视频。点击误差图或动作按钮可定位视频。</p>
<div class="grid"><div class="card"><b>120 秒</b>连续物理采样</div><div class="card"><b>29 段</b>参考动作与混合</div><div class="card"><b>0 次</b>初始化后的重置</div><div class="card"><b>待增强</b>全局位置精度与空翻</div></div>
<h2>实际数值与失败记录</h2><label>实验 <select id="mode"></select></label> <label>序列 <select id="seed"></select></label><div id="metrics"></div><svg id="plot" viewBox="0 0 1100 290" role="img" aria-label="固定世界坐标根 XY 误差随时间变化"></svg><p class="meta">硬失败为原门限：根高度误差 &gt;0.25m，或 Rzz 误差 &gt;0.8。为保留失败后的轨迹，诊断运行不执行自动终止/重置，仍逐帧独立计算并累计原门限失败。未把禁用终止调用当作没有失败。</p>
<h2>seed 0 动作时间轴</h2><div id="segments"></div>
<h2>复现边界</h2><ul><li>闭环动作由策略输出并推进 MuJoCo 3.11 / mjbatch 物理；不是把参考 qpos 写回机器人。时间步 5ms，控制周期 20ms；CPU FP32 推理，与原云端 mjwarp / BF16 是新的后端测试。</li><li>只有时间零点初始化。切换时只离线对参考做一次平移和朝向对齐，使用 0.6 秒五次平滑权重及四元数 SLERP。关节、位置先混合，再由原 MotionLib 统一差分计算速度。未来 10 帧窗口跨越所有内部边界。</li><li>原视频标注 ELF / EmbodiChain / Dexsim FastRT；视频画面不构成 SuperMimic 训练来源证明。本实验复现其演示形式与控制效果，机器人和引擎沿用已验收的 G1 / SuperMimic 体系。</li><li>固定序列种子 0、1、2 为适配训练参考；额外种子 3、4、5 在适配前冻结，作为序列外重复测试。它们仍来自原 LAFAN1 动作库，不是未见动作泛化证明。</li><li>零硬门限失败不等于无全局漂移。直接改变原策略未来目标输入的零样本试验已失败，全部失败数据保留在 goal-feedback 中，不作为改进方案。</li></ul>
<h2>空翻能力缺口</h2><p id="acro"></p><p>小型候选来源为 <a href="https://github.com/lez666/chunwan-flip">chunwan-flip</a> 提供的 G1 29DoF CSV。其 README 声明 MIT 仓库并说明数据源自 AMASS/CMU 重定向；这些来源说明被保留，不以仓库许可证替代原始数据权利说明。候选不是 BONES，也不声称与目标视频的具体动作相同。</p>
<h2>原始记录与复现命令</h2><p><a href="execution.json">执行台账</a> · <a href="runs/baseline/summary.json">基础版全部试验</a> · <a href="runs/baseline/trajectory.npz">基础版完整数值</a> · <a href="runs/goal-feedback/summary.json">失败的目标输入试验</a> · <a href="evidence/acro-candidate-audit.json">空翻来源与数据检查</a> · <a href="evidence/video-verification.json">完整视频解码校验</a> · <a href="../implementation/continuation-20261025-a/report/continuation-comparison.html">保留的旧基线报告</a></p>
<pre>.venv/Scripts/python.exe scripts/compose_sequences.py
.venv/Scripts/python.exe scripts/evaluate_continuous.py --mode baseline
.venv/Scripts/python.exe scripts/render_video.py --mode baseline --seed 0
.venv/Scripts/python.exe scripts/evaluate_acro.py</pre><details><summary>参考组成算法、对齐、速度与未来窗口</summary><pre id="manifest"></pre></details><details><summary>视频身份</summary><pre id="target"></pre></details></main>
<script id="data" type="application/json">DATA</script><script>
const D=JSON.parse(document.querySelector('#data').textContent),M=document.querySelector('#mode'),S=document.querySelector('#seed'),V=document.querySelector('#video'),P=document.querySelector('#plot');
const labels={'baseline':'原检查点：基础序列','baseline-heldout':'原检查点：额外序列','goal-feedback':'失败试验：直接修改目标输入'};
for(const name of Object.keys(D.modes)){let o=new Option(labels[name]||name,name);M.add(o)}M.value='baseline';
const fmt=(x,n=3)=>Number(x).toFixed(n);
function fillSeeds(){S.replaceChildren();for(const row of D.modes[M.value].series)S.add(new Option('seed '+row.metrics.seed,row.metrics.seed));draw()}
function draw(){const r=D.modes[M.value].series.find(x=>x.metrics.seed==S.value),m=r.metrics;document.querySelector('#metrics').innerHTML=`<table><tr><th>连续时长</th><th>重置</th><th>硬失败帧</th><th>XY RMSE / 峰值</th><th>平均全局 / 局部 body 误差</th><th>首个硬失败</th></tr><tr><td>${m.elapsed_seconds}s</td><td>${m.post_initialization_resets}</td><td>${m.hard_failure_frames}</td><td>${fmt(m.root_xy_rmse_m)} / ${fmt(m.root_xy_peak_m)} m</td><td>${fmt(m.body_global_mean_m)} / ${fmt(m.body_local_mean_m)} m</td><td>${m.first_hard_failure_s===null?'无':m.first_hard_failure_s+'s'}</td></tr></table>`;const ymax=Math.max(.1,...r.xy)*1.1;let g='';for(let i=0;i<5;i++){let y=240-i*50;g+=`<line x1="65" x2="1080" y1="${y}" y2="${y}" stroke="#e0e7ef"/><text x="8" y="${y+5}" fill="#526b82" font-size="14">${fmt(i*ymax/4,2)}</text>`}const pts=r.xy.map((x,i)=>`${65+i/(r.xy.length-1)*1015},${240-x/ymax*200}`).join(' ');P.innerHTML=g+`<text x="65" y="22" fill="#213c56" font-size="17">固定世界根 XY 误差（m） · ${M.value} / seed ${S.value}</text><polyline points="${pts}" fill="none" stroke="#b43d38" stroke-width="2.4"/><text x="65" y="272" font-size="14">0s</text><text x="560" y="272" font-size="14">60s</text><text x="1038" y="272" font-size="14">120s</text>`}
M.onchange=fillSeeds;S.onchange=draw;P.onclick=e=>{const b=P.getBoundingClientRect();V.currentTime=Math.max(0,Math.min(120,((e.clientX-b.left)/b.width*1100-65)/1015*120))};fillSeeds();
for(const s of D.segments){let b=document.createElement('button');b.textContent=`${s.segment}. ${s.source.replace('.csv','')} · ${(s.start_frame/50).toFixed(1)}s`;b.onclick=()=>V.currentTime=s.start_frame/50;document.querySelector('#segments').append(b)}
document.querySelector('#acro').textContent=`原策略在 ${D.acro.elapsed_seconds}s 的新参考测试中，于 ${D.acro.first_hard_failure_s}s 首次触发原硬失败门限，共 ${D.acro.hard_failure_frames} 帧失败；零重置。当前未通过空翻验收。`;
document.querySelector('#manifest').textContent=MANIFEST;document.querySelector('#target').textContent=TARGET;
</script></html>'''
    page=page.replace('>DATA</script>','>'+data+'</script>').replace('=MANIFEST;', '='+json.dumps(json.dumps(manifest,ensure_ascii=False,indent=2))+';').replace('=TARGET;','='+json.dumps(json.dumps(target,ensure_ascii=False,indent=2))+';')
    if 'baseline-heldout' in modes:
        trials=modes['baseline']['summary']['trials']+modes['baseline-heldout']['summary']['trials']
        ok=sum(t['passed_continuous_no_hard_failure'] for t in trials)
        page=page.replace('原 16000 检查点在 3 条预先固定的 120 秒 / 29 段序列上均为初始化后零重置、零原硬门限失败。',f'原 16000 检查点在全部 {len(trials)} 条预先固定的 120 秒 / 29 段序列中，{ok} 条未触发原硬失败门限，{len(trials)-ok} 条失败；所有试验初始化后均零重置。')
    comparisons=[]
    aggregate=[]
    for name,mode in modes.items():
        trials=mode['summary']['trials']
        row={'experiment':name,'seeds':[t['seed'] for t in trials],
             'hard_gate_passed':sum(t['passed_continuous_no_hard_failure'] for t in trials),
             'trials':len(trials),'mean_xy_rmse_m':float(np.mean([t['root_xy_rmse_m'] for t in trials])),
             'largest_xy_peak_m':max(t['root_xy_peak_m'] for t in trials),
             'resets':sum(t['post_initialization_resets'] for t in trials),
             'low_pelvis_entries':sum(t['floor_fall_entries'] for t in trials)}
        aggregate.append(row)
        comparisons.append(f'<tr><td><a href="runs/{html.escape(name)}/summary.json">{html.escape(name)}</a></td><td>{row["seeds"]}</td><td>{row["hard_gate_passed"]}/{row["trials"]}</td><td>{row["mean_xy_rmse_m"]:.3f}</td><td>{row["largest_xy_peak_m"]:.3f}</td><td>{row["resets"]}</td><td>{row["low_pelvis_entries"]}</td></tr>')
    experiment_table='<h2>全部实验对照</h2><div style="overflow:auto"><table><tr><th>实验 / 原始摘要</th><th>种子</th><th>原硬门限通过</th><th>平均 XY RMSE (m)</th><th>最大 XY 峰值 (m)</th><th>重置</th><th>低骨盆事件</th></tr>'+''.join(comparisons)+'</table></div><p class="meta">低骨盆事件仅计数骨盆低于 0.35m 的区间，蹲伏、跪地也会触发；它不是经过姿态或接触判定的跌倒次数。完整失败及切换窗口记录均在各行原始摘要中。</p>'
    page=page.replace('<h2>复现边界</h2>',experiment_table+'<h2>复现边界</h2>')
    page=page.replace('<h2>空翻能力缺口</h2>','<h2>训练与冻结审计</h2><p>位置观测由 99 维扩展到 101 维。初始策略动作一致性最大误差 2.75×10⁻⁶。完整模型的 300 次小批次试训使全部 6 条物理复测退步，保留为失败。adapter-150 降低 XY 漂移，但世界模型运行归一化统计仍改变，冻结审计未通过。adapter-600 修正该问题：后续 450 次更新的世界模型、价值网络及旧策略列逐张量冻结和保存读回均通过；这不追溯性地修复 adapter-150 的归一化改变。<a href="runs/global-adapter-600/proof.json">查看修正阶段证明</a>。</p><h2>空翻能力缺口</h2>')
    best='global-adapter-600-lookahead'
    if best in modes:
        trials=modes[best]['summary']['trials'];ok=sum(t['passed_continuous_no_hard_failure'] for t in trials)
        fresh=modes.get(best+'-fresh',{}).get('summary',{}).get('trials',[])
        extra=(f'另 {len(fresh)} 条在规划参数冻结后新生成的序列中，{sum(t["passed_continuous_no_hard_failure"] for t in fresh)} 条通过。' if fresh else '额外新序列复测待结束。')
        status=f'<div class="status"><b>基础连续动作已改进；空翻等完整目标尚未完成。</b><br>整链离线规划 + adapter-600：固定 {len(trials)} 条中 {ok} 条通过原硬门限，全部 120 秒 / 29 段 / 初始化后零重置；XY RMSE 范围 {min(t["root_xy_rmse_m"] for t in trials):.3f}–{max(t["root_xy_rmse_m"] for t in trials):.3f}m。{extra}原参考方案的失败全部保留；没有从不同方案挑选成功序列拼成通过率。</div>'
        page=re.sub(r'<div class="status">.*?</div>',lambda _:status,page,count=1,flags=re.S)
        page=page.replace("M.value='baseline';",f"M.value='{best}';")
    page=page.replace('<a href="../implementation/continuation-20261025-a/report/continuation-comparison.html">保留的旧基线报告</a>','旧基线报告保存在相邻 implementation/continuation-20261025-a/report 目录')
    # Select the improved video only after its complete decode receipt exists.
    verified=ROOT/'evidence/video-global-adapter-600-lookahead-seed0.json'
    if verified.exists() and json.loads(verified.read_text()).get('complete_decode_passed'):
        page=page.replace('src="videos/baseline-seed0.mp4"','src="videos/global-adapter-600-lookahead-seed0.mp4"')
        page=page.replace('视频固定为 baseline / seed 0','视频固定为 global-adapter-600-lookahead / seed 0')
        page=page.replace('href="evidence/video-verification.json"','href="evidence/video-global-adapter-600-lookahead-seed0.json"')
    commands='''已有参考、旧策略及失败记录均保留。复测请使用未使用过的 --mode 名称：
.venv/Scripts/python.exe deployment/reconstruct_global_policy.py
.venv/Scripts/python.exe scripts/evaluate_continuous.py --mode reproduction-new-01 --checkpoint policy/global-adapter-600-deployed.pt --reference-set references-lookahead --seeds 0 1 2 3 4 5
.venv/Scripts/python.exe scripts/render_video.py --mode reproduction-new-01 --seed 0
.venv/Scripts/python.exe scripts/evaluate_acro.py --mode acro-preserved-new-01
完整本机依赖版本见 evidence/local-runtime-versions.json。以上复用冻结参考，不覆盖已验收结果。'''
    page=re.sub(r'<pre>\.venv/Scripts/python\.exe scripts/compose_sequences\.py.*?</pre>',lambda _:'<pre>'+html.escape(commands)+'</pre>',page,count=1,flags=re.S)
    execution=json.loads((ROOT/'execution.json').read_text(encoding='utf-8'))
    cloud=execution.get('acro_specialist',{})
    progress=('单个后手翻片段已通过倒置与直立结束判据，完整轨迹已回收并校验' if cloud.get('full_physical_acceptance_verified') else '训练已结束，平台状态成功；完整轨迹回收和落地验收待核对' if cloud.get('platform_state')=='success' else cloud.get('status','尚未记录'))
    cloud_html='<h2>后手翻专项执行状态</h2><p>'+html.escape(progress)+'。任务 '+html.escape(cloud.get('job',''))+'；请求 1 张 H100，新增 2000 次更新，程序最长 7200 秒。训练进度不构成落地验收。</p>'
    cloud_html+='<p>该参考对应 <a href="https://mocap.cs.cmu.edu/search.php?subjectnumber=88">CMU subject 88 / motion 8</a>，原始说明为“crouch and flip backward on hands”，应称手撑地向后翻或后手翻类。参考中的手触地是预期接触，不能仅凭触手判失败。本轮只训练 88_08，不能推广为普通腾空后空翻或目标视频全部 90/180/360 空翻。</p>'
    acro_rows=[]
    for path in sorted((ROOT/'runs').glob('acro-*/summary.json')):
        trial=json.loads(path.read_text())
        if 'passed' not in trial:continue
        name=path.parent.name
        video=ROOT/'evidence'/f'video-{name}.json'
        link=(f' · <a href="videos/{html.escape(name)}.mp4">完整视频</a>' if video.exists() and json.loads(video.read_text()).get('complete_decode_passed') else '')
        acro_rows.append(f'<tr><td><a href="runs/{html.escape(name)}/summary.json">{html.escape(name)}</a>{link}</td><td>{trial["hard_failure_frames"]}</td><td>{trial.get("inversion_observed","未记录")}</td><td>{trial.get("landing_last_half_second_upright","未记录")}</td><td>{trial["post_initialization_resets"]}</td><td>{trial["passed"]}</td></tr>')
    if acro_rows:cloud_html+='<table><tr><th>实验 / 原始结果</th><th>原门限失败帧</th><th>倒置</th><th>末半秒直立</th><th>重置</th><th>通过</th></tr>'+''.join(acro_rows)+'</table>'
    contact_file=ROOT/'evidence/acro-h100-contact-audit.json'
    if contact_file.exists():
        contact=json.loads(contact_file.read_text());end=contact['end_state']
        cloud_html+=f'<p>单片段末半秒最低骨盆高度 {end["last_half_second_min_pelvis_m"]:.3f}m、最低 Rzz {end["last_half_second_min_Rzz"]:.3f}。从保留姿态重建的几何接触显示双腕在翻转阶段接触地面，未发现头/躯干触地；这是 20ms 采样的几何接触分析，不是接触力测量。结束根速度 {end["root_speed_m_s"]:.2f}m/s，不能称为已静止站稳。源参考存在脚部几何穿地，实际轨迹也有接触穿透；原硬门限通过不代表接触保真度通过。<a href="evidence/acro-h100-contact-audit.json">完整接触与结束状态审计</a>。</p>'
    acro_verified=ROOT/'evidence/video-acro-h100-2000-retest.json'
    if acro_verified.exists() and json.loads(acro_verified.read_text()).get('complete_decode_passed'):
        cloud_html+='<video controls preload="metadata" src="videos/acro-h100-2000-retest.mp4" aria-label="单个后手翻物理测试完整视频"></video><p class="meta">完整 4.72 秒测试；原始状态、归档和全部逐文件哈希均已核对。当前是单片段 99 维专项策略，尚不是全部新动作与旧库的统一跟踪策略。<a href="evidence/acro-local-transfer-verification.json">回收校验</a> · <a href="evidence/resource-final-20261027.json">本轮资源释放核验</a>。</p>'
    page=page.replace('当前未通过空翻验收。','这是保留的旧检查点失败记录；新专项结果见下表。')
    page=page.replace('<h2>原始记录与复现命令</h2>',cloud_html+'<h2>原始记录与复现命令</h2>')
    (ROOT/'evidence/experiment-comparison.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    (ROOT/'continuous-demo.html').write_text(page,encoding='utf-8')
    print('CONTINUOUS_REPORT_READY '+str(ROOT/'continuous-demo.html'))


if __name__=='__main__':main()
