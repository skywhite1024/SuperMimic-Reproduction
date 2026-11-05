"""Append an evidence-backed project review to the existing report, without rerunning experiments."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = PROJECT / 'research' / '20261101-report-refresh'
SOURCES = {}


def read(path, default=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return default
    raw = p.read_bytes()
    rel = p.relative_to(PROJECT).as_posix()
    SOURCES[rel] = {'path': rel, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw.decode('utf-8-sig'))


def href(path):
    p = Path(path)
    return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else '../' + p.relative_to(PROJECT).as_posix() if p.is_absolute() else str(path)


def library(mode):
    folder = ROOT / 'runs' / mode
    summary = read(folder / 'summary.json')
    if not summary:
        return None
    strict = read(folder / 'combined-assessment.json', {})
    joined = {x['clip']: x for x in strict.get('clips', [])}
    rows = []
    for x in summary.get('clips', []):
        if x.get('group') != 'new':
            continue
        c = joined.get(x['clip'], {})
        rows.append({'clip': x['clip'], 'score': x.get('norm_return_original'),
                     'hard': x.get('passed_original_hard_gate'), 'strict': c.get('combined_engineering_passed'),
                     'tracking': c.get('tracking_quality_passed'), 'contact': c.get('contact_review_passed'),
                     'xy': x.get('root_xy_rmse_m'), 'body': x.get('body_local_mean_m'),
                     'failed_checks': c.get('tracking_failed_checks', [])})
    return {'mode': mode, 'device': summary.get('device'), 'groups': summary.get('groups', {}),
            'strict_new': strict.get('combined_engineering_passed'), 'rows': rows,
            'source': f'runs/{mode}/summary.json', 'strict_source': f'runs/{mode}/combined-assessment.json' if strict else None}


def sequence(mode):
    d = read(f'runs/{mode}/continuous-assessment.json')
    if d is None:
        return None
    contact = read(f'runs/{mode}/continuous-contact-review.json')
    if contact:
        c = {x['seed']: x for x in contact.get('trials', [])}
        passed = sum(bool(x.get('continuous_engineering_passed')) and bool(c.get(x['seed'], {}).get('contact_review_passed')) for x in d['trials'])
        kind = '跟踪＋接触'
    else:
        passed = d.get('continuous_engineering_passed')
        kind = '历史连续跟踪合同，未与后续接触合同混同'
    return {'passed': passed, 'denominator': d.get('denominator', 3), 'kind': kind,
            'source': f'runs/{mode}/continuous-assessment.json'}


def compact_v2_library(x):
    return {'mode': x['mode'], 'device': x['device'], 'groups': x['groups'], 'strict_new': x['strict_new'],
            'source': x['summary_file'], 'strict_source': x['strict_file'],
            'rows': [{'clip': r['clip'], 'score': r.get('norm_return_original'), 'hard': r.get('passed_original_hard_gate'),
                      'strict': r.get('strict'), 'tracking': r.get('tracking_passed'), 'contact': r.get('contact_passed'),
                      'xy': r.get('root_xy_rmse_m'), 'body': r.get('body_local_mean_m'),
                      'failed_checks': r.get('tracking_failed_checks', [])} for r in x['rows'] if r.get('group') == 'new']}


def reference_key(variant):
    if variant == 'raw':
        return 'raw'
    if variant in {'cal', 'calibrated', 'corrected'}:
        return 'cal'
    raise ValueError(f'Unknown reference variant: {variant!r}')


def make_model(key, label, family, modes, sequence_modes=None, note='', report=None):
    rows = {}
    for device, variant, mode in modes:
        d = library(mode)
        if d:
            rows[f'{device}:{variant}'] = d
    sequences = {f'{dev}:{ref}': sequence(mode) for dev, ref, mode in (sequence_modes or [])}
    return {'key': key, 'label': label, 'family': family, 'runs': rows, 'sequences': sequences, 'note': note, 'report': report or 'unified-library-report.html#legacy-stages'}


def build_data():
    SOURCES.clear()
    source_audit = read(PROJECT / 'research/20261024/source-audit.json')
    publication = read(PROJECT / 'publication/PUBLICATION-COMPLETE.json')
    read(PROJECT / 'publication/retention1030-export-receipt.json')
    baseline = read(PROJECT / 'implementation/continuation-20261025-a/downloaded/evaluations/016000.json')
    confirm = read(PROJECT / 'implementation/continuation-20261025-a/downloaded/evaluations/016000-confirm.json')
    rollback = read(PROJECT / 'monitoring/rollback-checkpoint-review-20261101.json')
    specialist = read('evidence/specialist8807-1031d-final-assessment.json')
    qualified = read('evidence/qualified-expert-v2-1031-final-assessment.json')
    v2 = read('evidence/qualified-expert-v2-1031-comparison.json')
    oldw = read('evidence/qualified-expert-v3-oldw02-final-assessment1031.json')
    reproducibility = read('evidence/teacher8808-repro1031c-final-assessment.json')
    retention = read('evidence/retention-1030-physical-assessment.json')
    action_bound = read('evidence/action-bound-matched-comparison.json')
    manifest = read('library-manifest.json')
    descriptions = {x['name']: x for x in read('sources/motion-descriptions.json')}
    models = [
        make_model('baseline16', '16K 原基线', 'native', [('cpu', 'raw', 'old-16k-all55-v3')], note='99D；未对新15做严格合并验收。'),
        make_model('specialist18', '18K 后手翻专项', 'native', [('cpu', 'raw', 'parent-18k-all55-v3')], note='独立99D分支，不能替换统一模型。'),
        make_model('unified22', '22K 统一第一阶段', 'native', [('cuda','raw','unified4000-cuda-all55'), ('cpu','raw','unified4000-cpu-all55')],
                   [('cpu','raw','unified4000-frozen-continuous-v1'),('cpu','cal','unified4000-corrected-continuous-v1')], note='新增4000更新；早期连续合同有回溯应用，见源记录。'),
        make_model('unified26', '26K 统一第二阶段', 'native', [('cuda','raw','unified-stage2-raw-cuda-new15'),('cuda','cal','unified-stage2-corrected-cuda-all55'),('cpu','cal','unified-stage2-corrected-cpu-all55')],
                   [('cpu','raw','stage2-frozen-continuous-v1'),('cpu','cal','stage2-corrected-continuous-v1')], note='新动作训练参考做既定高度校准；原始参考独立复测。'),
        make_model('unified30', '30K 统一第三阶段', 'native', [('cuda','raw','unified-stage3-raw-cuda-new15'),('cuda','cal','unified-stage3-corrected-cuda-all55'),('cpu','raw','stage3-raw-cpu-new15-control'),('cpu','cal','unified-stage3-corrected-cpu-all55')],
                   [('cpu','raw','stage3-frozen-continuous-v1'),('cpu','cal','stage3-corrected-continuous-v1')], note='部分校准新动作改善，连续能力回退。'),
        make_model('parent30bound', '30K 父模型复评', 'bound20', [('cuda','raw','retention-v1-parent-bound20-raw-cuda-new15'),('cuda','cal','retention-v1-parent-bound20-corrected-cuda-all55'),('cpu','raw','stage3-actionbound20-raw-cpu-new15'),('cpu','cal','stage3-actionbound20-corrected-cpu-all55')],
                   [('cuda','raw','retention-v1-parent-bound20-raw-continuous-cuda'),('cuda','cal','retention-v1-parent-bound20-corrected-continuous-cuda'),('cpu','raw','stage3-actionbound20-frozen-continuous-v1'),('cpu','cal','stage3-actionbound20-corrected-continuous-v1')], report='action-bound-ablation-report.html', note='同权重、输出±20；不能与历史原输出视为同一条训练曲线。'),
        make_model('retention31', '31K Retention V1', 'bound20', [('cuda','raw','retention-v1-raw-cuda-new15'),('cuda','cal','retention-v1-corrected-cuda-all55'),('cpu','raw','retention-v1-raw-cpu-new15'),('cpu','cal','retention-v1-corrected-cpu-all55')],
                   [('cuda','raw','retention-v1-raw-continuous-cuda'),('cuda','cal','retention-v1-corrected-continuous-cuda'),('cpu','raw','retention-v1-raw-continuous-cpu'),('cpu','cal','retention-v1-corrected-continuous-cpu')], report='retention-1030-report.html', note='从30K独立分支；增加旧教师保持约束，候选未采用。'),
    ]
    q = {'key':'qualified31', 'label':'31K Qualified V2', 'family':'bound20', 'runs':{}, 'sequences':{},
         'note':'另一条从30K开始的1000更新分支，并非在Retention V1上继续；候选未采用。', 'report':'qualified-expert-v2-1031-report.html'}
    for x in v2['libraries']:
        if x['model']=='candidate':
            q['runs'][f"{x['device']}:{reference_key(x['variant'])}"] = compact_v2_library(x)
    for x in v2['continuous']:
        if x['model']=='candidate':
            q['sequences'][f"{x['device']}:{reference_key(x['variant'])}"] = {'passed':x['joint_passed'],'denominator':x['denominator'],'kind':'跟踪＋接触','source':x['assessment_file']}
    models.append(q)
    # A raw-only new15 evaluation has no old40 group. Retain the paired same-model all55
    # source explicitly; do not silently present it as measured in the raw-only run.
    for m in models:
        for key, r in m['runs'].items():
            dev=key.split(':')[0]
            old=r['groups'].get('lafan')
            old_source=r['source']
            if old is None:
                paired=m['runs'].get(f'{dev}:cal')
                if paired and paired['groups'].get('lafan'):
                    old=paired['groups']['lafan'];old_source=paired['source']
            r['old_group']=old;r['old_source']=old_source if old else None
    # Source-control metadata only; no tensor loading, model selection or training here.
    checkpoint_rows=[]
    for m in rollback['models']:
        p=PROJECT / m['path']
        checkpoint_rows.append({'label':m['label'],'path':'../'+m['path'],'present_now':p.is_file(),
                                'bytes':p.stat().st_size if p.is_file() else None,
                                'load_verified_in_review':m.get('load_verified'),
                                'full_training_state':m.get('full_training_state_present'), 'iteration':m.get('iteration'),
                                'observation_dim':m.get('observation_dim')})
    proof_names=[('统一第一阶段','unified-h100-4000'),('统一第二阶段','unified-h100-stage2-4000'),('统一第三阶段','unified-h100-stage3-4000'),('Retention V1','unified-retention-v1-h100-1000'),('Qualified V2','unified-qualified-expert-v2-h100-1000'),('88_07专项','specialist8807-h100-1000-1031d')]
    training=[]
    for label, folder in proof_names:
        d=read(f'runs/{folder}/proof.json', {})
        training.append({'label':label,'updates':d.get('new_updates',d.get('new_iterations')),
                         'seconds':d.get('wall_seconds'), 'source':f'runs/{folder}/proof.json'})
    timeline=[
        {'date':'10/24–26','tag':'基线','title':'从社区代码到可恢复的LAFAN1基线','status':'已验证','text':'12K：回报0.802814、3/40提前失败。受控续训到16K：两次宏平均0.847492和0.847733，均0/40。它是社区基线，不是论文全部结果。','link':'../implementation/continuation-20261025-a/report/continuation-comparison.html'},
        {'date':'10/26','tag':'范围','title':'BONES-SEED准备后暂停','status':'已暂停','text':'做过数据、依赖和分布式路线审计；未取得论文129,514条规模复现结果。后续重点转向用户视频中的连续动作效果。','link':'../bones-seed-20261026-a/report/preparation-status.html'},
        {'date':'10/27','tag':'连续','title':'先把40条独立评测变成连续切换','status':'部分达成','text':'原基线6条120秒序列，5条未触发原门限但有米级漂移。整体300步微调退步；受限XY输入适配加离线选段后，12条基础序列通过门限、零重置。只覆盖走跑舞蹈跳跃，未证明高动态连续控制。','link':'../interactive-g1-20261027-a/continuous-demo.html'},
        {'date':'10/27','tag':'专项','title':'88_08后手翻：学到单段，出现严重遗忘','status':'有限通过','text':'单H100补训2000更新。4.72秒倒置和末段直立测试通过；不是腾空后空翻。专项副本的旧40在同协议CPU回测降到2/40，原16K文件保留。','link':'../interactive-g1-20261027-a/runs/acro-h100-2000-retest/summary.json'},
        {'date':'10/27–28','tag':'统一','title':'15条新素材与40条旧动作，三个4000更新阶段','status':'目标未达','text':'22K、26K、30K各自有完整数值与失败尾段。新动作、旧动作和连续能力没有共同单调提升；原始/高度校准参考分开记录。','link':'#legacy-stages'},
        {'date':'10/28–30','tag':'诊断','title':'同权重动作限幅对照','status':'完成诊断','text':'发现采集/想象/部署的输出限幅不一致，做±20部署对照，新增训练0次。初次失败之后才出现的越界不能解释首次失败，不能直接认定为训练失败的唯一原因。','link':'action-bound-ablation-report.html'},
        {'date':'10/29–30','tag':'归档','title':'公开代码、推理权重与报告快照','status':'历史发布','text':'10月29日完成v0.1.0代码、6个Release资产及报告页面的发布核验；10月30日另准备增量导出。历史发布不代表后续实验已经同步，本次仅更新本地总报告。','link':'../publication/PUBLICATION-COMPLETE.json'},
        {'date':'10/30','tag':'统一','title':'Retention V1：用旧教师缓解遗忘','status':'不采用','text':'1000新增更新；CUDA原始/校准新15严格7/15、9/15，旧40为37/40，连续两组均0/3；未满足采用条件。','link':'retention-1030-report.html'},
        {'date':'10/30','tag':'诊断','title':'先检验“专家”能否成为合格教师','status':'完成诊断','text':'统一旧教师与新技能教师分开审查，88_07、88_09仍缺可靠技能。教师仅在训练时使用，不等于部署多个专家。','link':'teacher-qualification-1030-report.html'},
        {'date':'10/31','tag':'统一','title':'Qualified V2：部分连续收益与旧能力回退并存','status':'不采用','text':'1000更新；CUDA新15严格9/15原始、10/15校准，旧40由39降到38；原始与校准连续均从0/3到1/3。CPU结果不同，不能混成一个分数。','link':'qualified-expert-v2-1031-report.html'},
        {'date':'10/31','tag':'诊断','title':'旧教师权重0.1→0.2对照没有进入训练','status':'0更新','text':'原始教师穿地53.44mm超过固定50mm；另一评测46.81mm通过。训练前门槛已停止，无法评价0.2权重有效性。88_08原始/校准文件实际相同，差异不能算作校准收益。','link':'qualified-expert-v3-oldw02-1031-report.html'},
        {'date':'10/31','tag':'诊断','title':'同输入三次新进程：教师通过结论波动','status':'2/3通过','text':'最大穿地30.87、46.22、51.37mm，跨越50mm门槛。首动作等记录输入相同，状态在第一次env.step后分化。样本不足以估计成功率，分化的具体内部原因未定位。','link':'teacher8808-repro1031c-report.html'},
        {'date':'10/31–11/01','tag':'专项','title':'88_07侧手翻专项：姿态改善，位置与接触未达标','status':'不采用','text':'从原16K独立补训1000更新，局部姿态误差均值0.509→0.158m，XY误差0.720→0.770m。CPU/CUDA原始/校准严格均0/3；专项副本旧40降到0/40，不能作为合格教师。','link':'specialist8807-1031d-report.html'},
        {'date':'11/01','tag':'审计','title':'回退候选与模型保存审计','status':'仅审查','text':'9个关键模型已有本地加载核验。保留16K可靠基线，22K作为优先回退候选，26K/30K保留作对照；尚未执行回退，也未据此开新训练。','link':'../monitoring/rollback-checkpoint-review-20261101.json'},
    ]
    infra=[
        {'name':'权重/源码下载超时','updates':0,'meaning':'传输和初始化失败，不能评价算法','source':'evidence/qualified-expert-v2-1031a-final-closure.json'},
        {'name':'CRLF命令参数导致argparse失败','updates':0,'meaning':'科学训练启动之前退出；改LF副本后恢复','source':'evidence/qualified-expert-v3-afs-scientific-command-CRLF-diagnosis.json'},
        {'name':'0.2权重教师门槛停止','updates':0,'meaning':'科学前置门槛未过，权重假设尚未检验','source':'evidence/qualified-expert-v3-oldw02-final-assessment1031.json'},
        {'name':'登录过期与结果回收中断','updates':None,'meaning':'88_07云端训练和评测正常完成；中断发生于查看与回收','source':'evidence/specialist8807-1031d-final-resource-verification.json'},
    ]
    for x in infra:read(x['source'])
    # Data for the last specialist error chart comes from the retained per-trial metrics.
    specialist_means={}
    for model in ['parent','candidate']:
        vals=[]
        for ref in ['raw','corrected']:
            for repeat in range(3):
                d=read(f'runs/specialist8807-1031d-{model}-{ref}-cuda-r{repeat}/summary.json',{})
                vals.extend(d.get('clips',[]))
        specialist_means[model]={key:sum(v[key] for v in vals if v.get(key) is not None)/sum(v.get(key) is not None for v in vals) if any(v.get(key) is not None for v in vals) else None for key in ['body_local_mean_m','root_xy_rmse_m']}
    videos=[
        {'title':'基础动作连续120秒','scope':'12条基础序列通过原门限；不包含新高动态库','file':'../interactive-g1-20261027-a/videos/global-adapter-600-lookahead-seed0.mp4'},
        {'title':'88_08手撑地后翻','scope':'单段有限验收；接触重复稳定性后来发现不足','file':'../interactive-g1-20261027-a/videos/acro-h100-2000-retest.mp4'},
        {'title':'Qualified V2：通过的连续seed0','scope':'CUDA原始参考；本组仅1/3严格通过','file':'videos/qualified-expert-v2-1031-raw-sequences-cuda/seed-0.mp4'},
        {'title':'Qualified V2：保留失败尾段seed2','scope':'完整120秒失败过程；零重置不等于成功','file':'videos/qualified-expert-v2-1031-raw-sequences-cuda/seed-2.mp4'},
        {'title':'88_07专项训练前','scope':'CUDA固定重复r0；原始参考','file':'videos/specialist8807-1031d-parent-raw-cuda-r0/88_07_stageii_unitree_g1.mp4'},
        {'title':'88_07专项训练后','scope':'姿态改善，但位移/接触未通过；相同参考','file':'videos/specialist8807-1031d-candidate-raw-cuda-r0/88_07_stageii_unitree_g1.mp4'},
    ]
    for i,x in enumerate(videos,1):
        x['exists']=(ROOT/x['file']).is_file()
        poster=f'posters/journal-{i}.jpg'
        x['poster']=poster if (ROOT/poster).is_file() else None
    reports=[('原LAFAN1受控续训','../implementation/continuation-20261025-a/report/continuation-comparison.html'),('连续基础与88_08','../interactive-g1-20261027-a/continuous-demo.html'),('同权重限幅','action-bound-ablation-report.html'),('Retention V1','retention-1030-report.html'),('教师资格','teacher-qualification-1030-report.html'),('Qualified V2','qualified-expert-v2-1031-report.html'),('0.2权重0更新','qualified-expert-v3-oldw02-1031-report.html'),('教师重复性','teacher8808-repro1031c-report.html'),('88_07专项与24段视频','specialist8807-1031d-report.html')]
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    for rel in ['implementation/uni/README_zh.md',
                'implementation/uni/src/unisupermimic/rl/algos/supermimic/learner.py',
                'implementation/uni/src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py',
                'implementation/uni/src/unisupermimic/conf/supermimic/config.yaml']:
        p=PROJECT/rel
        raw=p.read_bytes()
        SOURCES[rel]={'path':rel,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    return {'updated_at':now,'evidence_cutoff':'2026-11-01，最新完成88_07专项收尾及检查点审查',
            'community_audit':source_audit,'historical_publication':{'repository':publication['repository'],'release':publication['release'],'pages':publication['pages'],'at':publication['at']},
            'full_goal_complete':False,'baseline':{'primary':baseline['norm_return'],'confirm':confirm['norm_return'],'failed':baseline['n_failed']},
            'models':models,'timeline':timeline,'training':training,'infra':infra,
            'library':[{'clip':Path(x['path']).stem,'description':descriptions.get(Path(x['path']).stem,{}).get('description','待标注'),'source':x['source_url']} for x in manifest['files']],
            'qualified_metrics':qualified['metrics'],'retention_counts':retention['counts'],
            'teacher_repro':reproducibility,'old_weight_attempt':{'updates':oldw['new_training_updates'],'raw_mm':-oldw['teacher_gates']['raw']['actual_min_after_initial_100ms_m']*1000,'cal_mm':-oldw['teacher_gates']['corrected']['actual_min_after_initial_100ms_m']*1000},
            'specialist_means':specialist_means,'specialist_counts':specialist['counts'],'specialist_old':{'parent':specialist['boundary']['parent']['old_hard'],'candidate':specialist['boundary']['candidate']['old_hard']},
            'specialist_videos':specialist['complete_videos'],'checkpoints':checkpoint_rows,
            'rollback_scope':rollback['scope'],'videos':videos,'reports':[{'title':x,'path':y} for x,y in reports],
            'sources':list(SOURCES.values()),'source_chats':[{'title':'调研 SuperMimic 社区复现','id':'01a16e2c-117e-7d92-ba17-a90454c49c06'},{'title':'复现 SuperMimic','id':'01a16e3e-b82d-7242-ae63-c7fa6dca7faf'}]}


def update_report():
    AUDIT.mkdir(parents=True,exist_ok=True)
    data=build_data()
    (ROOT/'project-journal-20261101.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=ROOT/'unified-library-report.html'
    original=path.read_text(encoding='utf-8-sig')
    backup=AUDIT/'unified-library-report.before-journal.html'
    if not backup.exists():backup.write_text(original,encoding='utf-8')
    current_base=AUDIT/'unified-library-report.journal-base-current.html'
    start='<!-- PROJECT-JOURNAL-20261101-START -->'
    end='<!-- PROJECT-JOURNAL-20261101-END -->'
    if start in original:
        # Rebuild from the stable pre-journal report so repeated runs do not nest wrappers.
        original=(current_base if current_base.exists() else backup).read_text(encoding='utf-8')
    else:
        # A fresh legacy rebuild may contain newer experiments. Preserve that base
        # separately from the immutable first backup for subsequent journal refreshes.
        current_base.write_text(original,encoding='utf-8')
    template=(Path(__file__).parent/'project_journal_template.html').read_text(encoding='utf-8')
    fragment=template.replace('__JOURNAL_DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    original=re.sub(r'<title>.*?</title>','<title>SuperMimic 复现与探索总览 · 截至2026-11-01</title>',original,count=1,flags=re.S)
    original=original.replace('<main>','<main>'+start+fragment+end+'<details id="legacy-stages" class="jr-legacy"><summary>展开原三阶段完整账本、逐动作记录与旧训练曲线</summary>',1)
    pos=original.rfind('</main>')
    original=original[:pos]+'</details>'+original[pos:]
    path.write_text(original,encoding='utf-8')
    (AUDIT/'journal-build-receipt.json').write_text(json.dumps({'report':str(path),'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'backup':str(backup),'source_files':len(SOURCES),'models':len(data['models']),'new_clip_denominator':len(data['library']),'full_goal_complete':False,'training_or_cloud_actions_performed':False},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'updated':str(path),'sources':len(SOURCES),'timeline':len(data['timeline']),'models':len(data['models']),'videos_found':sum(x['exists'] for x in data['videos'])},ensure_ascii=False))


if __name__=='__main__':
    update_report()
