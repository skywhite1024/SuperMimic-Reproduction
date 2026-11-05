"""Build the finite specialist report from verified physical results, never training loss alone."""
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABEL = 'specialist8807-1031d'


def load(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))


def esc(value):
    return html.escape(str(value), quote=True)


def main():
    result = load(f'downloaded/{LABEL}/recovered.json')
    transfer = load(f'evidence/{LABEL}-local-transfer-verification.json')
    resource = load(f'evidence/{LABEL}-final-resource-verification.json')
    proof = load('runs/specialist8807-h100-1000-1031d/proof.json')
    assert result['new_training_updates'] == proof['new_iterations'] == 1000
    rows = []
    videos = []
    for device in ['cuda', 'cpu']:
        for role in ['parent', 'candidate']:
            for variant in ['raw', 'corrected']:
                for repetition in range(3):
                    mode = f'{LABEL}-{role}-{variant}-{device}-r{repetition}'
                    row = load(f'runs/{mode}/specialist-assessment.json')
                    row.update(role=role, device=device, variant=variant, repetition=repetition)
                    rows.append(row)
                    video = load(f'videos/{mode}/88_07_stageii_unitree_g1.json')
                    assert video['complete_decode_passed']
                    assert hashlib.sha256((ROOT/video['file']).read_bytes()).hexdigest() == video['sha256']
                    videos.append(video)
    counts = {device: {role: {variant: sum(x['strict_passed'] for x in rows
        if (x['device'], x['role'], x['variant']) == (device, role, variant))
        for variant in ['raw', 'corrected']} for role in ['parent', 'candidate']}
        for device in ['cuda', 'cpu']}
    boundary = {}
    for role in ['parent', 'candidate']:
        mode = f'{LABEL}-{role}-corrected-all55-cuda'
        if (ROOT/f'runs/{mode}/combined-assessment.json').exists():
            summary = load(f'runs/{mode}/summary.json')
            combined = load(f'runs/{mode}/combined-assessment.json')
            boundary[role] = {'new_strict': combined['combined_engineering_passed'],
                'new_hard': summary['groups']['new']['passed'],
                'old_hard': summary['groups']['lafan']['passed'],
                'old_macro_return': summary['groups']['lafan']['macro_return_original'],
                'old_failed_clips': [x['clip'] for x in summary['clips']
                                    if x['group'] == 'lafan' and not x['passed_original_hard_gate']],
                'summary': summary, 'combined': combined}
        else:
            boundary[role] = result['boundary'][role]
    delta = {}
    if all('old_hard' in boundary[x] for x in ['parent', 'candidate']):
        for group in ['new', 'lafan']:
            def passed(role):
                if group == 'new':
                    return {x['clip'] for x in boundary[role]['combined']['clips']
                            if x['combined_engineering_passed']}
                return {x['clip'] for x in boundary[role]['summary']['clips']
                        if x['group'] == group and x['passed_original_hard_gate']}
            before, after = passed('parent'), passed('candidate')
            delta[group] = {'gained': sorted(after-before), 'lost': sorted(before-after)}
    candidate_passed_all = all(counts[d]['candidate'][v] == 3
                              for d in ['cuda', 'cpu'] for v in ['raw', 'corrected'])
    assessment = {'status': 'finite_specialist_phase_complete', 'new_training_updates': 1000,
        'counts': counts, 'boundary': boundary, 'matched_library_delta': delta,
        'candidate_fixed_target_acceptance_all_passed': candidate_passed_all,
        'expert_automatically_promoted_to_teacher': False,
        'unified101D_candidate_adopted': False, 'full_goal_complete': False,
        'all_24_fixed_repetitions_retained': True, 'complete_videos': len(videos),
        'raw_and_calibrated_target_identical': True, 'target_constant_z_shift_m': 0.0,
        'physical_window_seconds': 2.52, 'contact_force_measured': False,
        'training_proof': proof, 'resource': resource, 'transfer': transfer}
    (ROOT/f'evidence/{LABEL}-final-assessment.json').write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    labels = [f'{d}\n{r}' for d in ['CUDA', 'CPU'] for r in ['parent', 'candidate']]
    groups = [[x for x in rows if x['device'] == d and x['role'] == r]
              for d in ['cuda', 'cpu'] for r in ['parent', 'candidate']]
    for axis, key, title in zip(axes, ['root_xy_rmse_m', 'body_local_mean_m', 'penetration_mm'],
                                ['Root XY RMSE (m)', 'Local body error (m)', 'Worst penetration (mm)']):
        values = [[max(0, -1000*x['actual_min_after_initial_100ms_m']) if key == 'penetration_mm'
                   else x['tracking'][key] for x in group] for group in groups]
        mean = np.array([np.mean(x) for x in values])
        error = np.array([[mean[i]-min(x) for i, x in enumerate(values)],
                          [max(x)-mean[i] for i, x in enumerate(values)]])
        axis.bar(labels, mean, yerr=error, color=['#718096', '#2463ad']*2, capsize=5)
        axis.set_title(title)
        axis.grid(axis='y', alpha=.2)
        if key == 'penetration_mm':
            axis.axhline(50, color='#bc3048', linestyle='--', label='50 mm limit')
            axis.legend()
    fig.suptitle('Fixed target controls: means and full range across all retained repetitions')
    comparison = ROOT/f'evidence/{LABEL}-comparison.png'
    fig.savefig(comparison, dpi=160)
    fig.savefig(comparison.with_suffix('.svg'))
    plt.close(fig)

    fields = ['iteration','wall_h','sim_steps','wm_loss','wm_err8','pi_loss','v_loss','imag_rew',
              'pi_gradnorm','imag_survival','sim_rew','ep_len','ep_ret_norm','term_frac','it_per_s','lr']
    csv_path = ROOT/'runs/specialist8807-h100-1000-1031d/train_log.csv'
    raw = list(csv.reader(csv_path.read_text().splitlines()))
    header = raw[0] if raw[0][0] == 'iteration' else fields
    data = raw[1:] if raw[0][0] == 'iteration' else raw
    assert header == fields and all(len(row) == len(fields) for row in data)
    records = [{key: float(value) for key, value in zip(fields, row)} for row in data]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for axis, key in zip(axes.flat, ['wm_loss', 'wm_err8', 'ep_ret_norm', 'pi_loss']):
        axis.plot([x['iteration']-16000 for x in records], [x[key] for x in records], marker='.')
        axis.axvline(400, color='#8b5b21', linestyle='--', label='WM-only warmup ends')
        axis.set(title=key, xlabel='New updates')
        axis.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle('Native training logs; training episode resets allowed, separate from fixed evaluation')
    curves = ROOT/f'evidence/{LABEL}-training-curves.png'
    fig.savefig(curves, dpi=160)
    fig.savefig(curves.with_suffix('.svg'))
    plt.close(fig)

    verdict = '目标固定验收全部通过，后续仍需独立评估能否用于统一合并。' if candidate_passed_all else '本轮专家尚未通过全部固定目标验收，不能提升为合格教师或统一模型。'
    old_boundary_text = ''
    if all('old_hard' in boundary[x] for x in ['parent', 'candidate']):
        old_boundary_text = f'<p><strong>独立分支的旧40原硬门槛：{boundary["parent"]["old_hard"]}/40 → {boundary["candidate"]["old_hard"]}/40；新15严格门槛：{boundary["parent"]["new_strict"]}/15 → {boundary["candidate"]["new_strict"]}/15。候选不作为统一模型采用。</strong></p>'
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>88_07 专项训练 · 1031d</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#172436;background:#f4f7fb;margin:0}main{max-width:1280px;margin:auto;padding:32px}h1{font-size:36px}section,article{background:white;border:1px solid #dce4ee;border-radius:12px;padding:22px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #dce4ee}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}video{width:100%;max-width:720px;background:#162132}img{max-width:100%}.scroll{overflow-x:auto}select{padding:8px;margin:5px}.fail{color:#a52b3d}.pass{color:#19734b}[hidden]{display:none!important}</style><main>',
        '<p>SuperMimic 社区实现 · 独立 99D 专家 · 有限实验 1031d</p><h1>88_07 专项：训练、固定验收与遗忘边界</h1>',
        f'<section><p><strong>{verdict}</strong></p>{old_boundary_text}<p>已验证 1000 次新增更新：400 次世界模型预热＋600 次联合学习；16K → 17K。训练使用原生奖励、物理与参考，归一化固定。最终一个 101D 策略覆盖新15＋旧40及连续控制的目标仍未完成。</p><p>1张H100／4CPU／64GiB／1副本／0重启；程序总上限5400秒，初始化上限600秒。训练证明记录 {proof["wall_seconds"]:.2f} 秒，含初始化、固定评估与归档的完整程序 {result["elapsed_seconds"]:.2f} 秒。完整检查点、策略导出与功能回读通过。</p><p>作业于2026年10月31日23:02:07正常结束。22:44左右登录过期中断的是助手查看与回收，云端继续完成了全部预定阶段；59个阶段退出0。本次收尾没有重跑训练或GPU评测。</p><p>候选CUDA目标的原始硬门槛为6/6通过，但严格验收仍为0/6。局部姿态误差与朝向改善，不能抵消全局位移、回报和意外躯干接触未达标。</p></section>',
        '<section><h2>固定目标结果</h2><p>每个后端、模型和参考标签均固定3次，seed=0、独立进程。126个控制步／2.52秒，无初始化后重置或机器人状态写入，全部失败尾部保留。88_07校准高度偏移为0，原始和校准文件逐字节相同；两组标签不构成不同参考输入。</p><table><tr><th>后端</th><th>模型</th><th>原始严格通过</th><th>校准严格通过</th></tr>']
    for device in ['cuda', 'cpu']:
        for role in ['parent', 'candidate']:
            parts.append(f'<tr><td>{device.upper()}</td><td>{role}</td><td>{counts[device][role]["raw"]}/3</td><td>{counts[device][role]["corrected"]}/3</td></tr>')
    parts += [f'</table><img src="evidence/{comparison.name}" alt="完整重复范围的误差和穿地比较"><p>接触为几何审查，未测接触力。严格验收包含原跟踪质量、初始100ms之后穿地≤50mm和意外头／躯干／骨盆接触样本0。均值和范围完整保留，不能从训练损失推断动作成功。</p></section>',
        f'<section><h2>训练曲线</h2><img src="evidence/{curves.name}" alt="原生训练日志曲线"><p>训练允许原生重置，训练回报不能替代2.52秒无重置的独立评测。联合阶段中，世界模型误差先升后降，记录的训练回报下降；不能据策略损失下降推断真实能力持续改善，也未由此确定本轮失败的因果机制。</p><details><summary>全部原生训练日志点</summary><pre>'+esc(json.dumps(records,indent=2))+'</pre></details></section>',
        '<section><h2>库能力边界：CUDA 校准55</h2><p>以下仅是独立99D专家的能力边界；新15使用跟踪＋接触严格门槛，旧40为既有原始硬门槛，两种口径分别报告。此处不是最终101D统一模型或连续验收。两套55的全部失败轨迹均已回收保留；本页24段视频覆盖目标动作全部固定重复，未将旧40视频称为已完成。</p><pre>'+esc(json.dumps({role: {k:v for k,v in row.items() if k not in ['summary','combined']} for role,row in boundary.items()},ensure_ascii=False,indent=2))+'</pre><h3>同后端匹配变化</h3><pre>'+esc(json.dumps(delta,ensure_ascii=False,indent=2))+'</pre><details><summary>两套55全部逐动作数值与失败项</summary><pre>'+esc(json.dumps(boundary,ensure_ascii=False,indent=2))+'</pre></details></section>',
        '<section><h2>下一项可检验假设，尚未执行</h2><p>优先检验全局位移观测不足：保持原奖励、物理、参考和门槛，只加入全局XY误差观测，检验能否降低漂移。本轮结果支持把它作为候选解释，尚未建立因果。现有101D实现同时加入XY奖励，不能直接复用后宣称只改了观测。原16K、30K统一模型及历史模型保留，本轮不自动追加同配置训练。</p></section>',
        '<section><h2>全部24次目标轨迹与视频</h2><p>灰色为真实仿真轨迹，绿色参考仅显示时向Y平移1.5米；数值使用未平移世界坐标。每段视频完整解码并验证哈希。</p><label>后端 <select id="device"><option value="">全部</option><option>cuda</option><option>cpu</option></select></label><label>模型 <select id="role"><option value="">全部</option><option>parent</option><option>candidate</option></select></label><label>参考 <select id="variant"><option value="">全部</option><option>raw</option><option>corrected</option></select></label></section>']
    for row, video in zip(rows, videos):
        t = row['tracking']
        passed = row['strict_passed']
        parts.append(f'<article class="run" data-device="{row["device"]}" data-role="{row["role"]}" data-variant="{row["variant"]}"><h3>{esc(row["run"])}</h3><p class="{"pass" if passed else "fail"}">严格验收：{"PASS" if passed else "FAIL"} · XY RMSE {t["root_xy_rmse_m"]:.4f} m · 局部姿态误差 {t["body_local_mean_m"]:.4f} m · 最大穿地 {max(0,-1000*row["actual_min_after_initial_100ms_m"]):.3f} mm · 意外躯干接触 {row["unexpected_trunk_contact_samples"]}</p><video controls preload="none" src="{esc(video["file"])}"></video><details><summary>完整数值、失败项与视频哈希</summary><pre>{esc(json.dumps({"assessment":row,"video":video},ensure_ascii=False,indent=2))}</pre></details></article>')
    parts += ['<section><h2>复核入口与资源终态</h2><p><a href="evidence/'+LABEL+'-final-assessment.json">完整实验账本</a> · <a href="evidence/'+LABEL+'-local-transfer-verification.json">归档及逐文件核验</a> · <a href="policy/specialist8807-h100-1000-1031d.pt">策略导出</a> · <a href="runs/specialist8807-h100-1000-1031d/final-full.pt">完整检查点</a></p><pre>'+esc(json.dumps(resource,ensure_ascii=False,indent=2))+'</pre><p>报告数据与本地链接通过静态核验；视频逐帧解码通过。内置浏览器文件预览受限，未宣称完成浏览器交互验收。BONES路线继续暂停。本轮不自动追加相同配置训练块。</p></section>',
        '<script>function filter(){document.querySelectorAll(".run").forEach(el=>{el.hidden=["device","role","variant"].some(k=>document.getElementById(k).value&&el.dataset[k]!==document.getElementById(k).value)})}document.querySelectorAll("select").forEach(el=>el.addEventListener("change",filter))</script></main></html>']
    report = ROOT/f'{LABEL}-report.html'
    report.write_text('\n'.join(parts), encoding='utf8')
    print(json.dumps({'report':str(report),'counts':counts,'videos':len(videos),
                      'boundary_delta':delta,'full_goal_complete':False},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
