import json
import pytest
import torch

from data import CandleWindows, Standardizer, loader
from model import seq2seq_model
from output import create_run, GradientHistory
from training import (calibrate, calibrate_temporary, final_stage_durations, fit,
                      global_gradient_norm, losses, seed_everything, train_epoch)
from transforms import reconstruct_ohlcv


@pytest.mark.parametrize('experiment',['2A','2B'])
def test_shapes_gradients_losses_and_valid_candles(frame,cfg,experiment):
    dataset=CandleWindows(frame,0,600,cfg,1)
    batch=next(iter(loader(dataset,cfg)))
    seed_everything(cfg.seed)
    model=seq2seq_model(cfg,dataset.scaler,experiment)
    out=model(batch['context'],batch['target'],teacher_forcing=.5,generator=torch.Generator().manual_seed(12))
    assert out['predictions'].shape==(cfg.batch_size,12,5)
    assert out['attention'].shape==(cfg.batch_size,12,48)
    torch.testing.assert_close(out['attention'].sum(-1),torch.ones(cfg.batch_size,12))
    assert (out['attention']>=0).all()
    raw=reconstruct_ohlcv(model.physical(out['predictions']).double(),batch['reference_close'])
    assert (raw[:,:,1]>=torch.maximum(raw[:,:,0],raw[:,:,3])).all()
    assert (raw[:,:,2]<=torch.minimum(raw[:,:,0],raw[:,:,3])).all()
    assert (raw[:,:,4]>=0).all()
    components=losses(model,out,batch,cfg,{'return':.5,'trend':.25})
    if experiment=='2A':
        assert out['trend_logits'] is None and components['return']==0 and components['trend']==0
        assert components['total'] is components['base']
    else:
        assert out['trend_logits'].shape==(cfg.batch_size,12,3)
        torch.testing.assert_close(components['total'],components['base']+.5*components['return']+.25*components['trend'])
    components['total'].backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.attention.energy.weight.grad.abs().sum()>0
    model.eval()
    with pytest.raises(ValueError,match='training'):model(batch['context'],batch['target'],teacher_forcing=1)


def test_shared_initialization_and_rng_unchanged(frame,cfg):
    dataset=CandleWindows(frame,0,600,cfg,1)
    seed_everything(cfg.seed); a=seq2seq_model(cfg,dataset.scaler,'2A'); rng_a=torch.get_rng_state()
    # Caller draws no longer influence shared initialization, and constructors
    # preserve caller CPU RNG entirely.
    torch.testing.assert_close(rng_a, torch.Generator().manual_seed(cfg.seed).get_state())
    torch.rand(37); caller_rng=torch.get_rng_state()
    b=seq2seq_model(cfg,dataset.scaler,'2B'); rng_b=torch.get_rng_state()
    for name,p in a.named_parameters():torch.testing.assert_close(p,dict(b.named_parameters())[name])
    torch.testing.assert_close(caller_rng,rng_b)
    a.eval();b.eval();context=torch.stack([dataset[0]['context']])
    torch.testing.assert_close(a(context)['predictions'],b(context)['predictions'])
    cfg.head_seed_offset+=1;c=seq2seq_model(cfg,dataset.scaler,'2B')
    for name,p in a.named_parameters():torch.testing.assert_close(p,dict(c.named_parameters())[name])
    assert not torch.equal(b.trend_head[0].weight,c.trend_head[0].weight)


def test_return_loss_backpropagates(frame,cfg):
    ds=CandleWindows(frame,0,600,cfg,1);batch=next(iter(loader(ds,cfg)))
    model=seq2seq_model(cfg,ds.scaler,'2B');out=model(batch['context'])
    losses(model,out,batch,cfg,{'return':1,'trend':1})['return'].backward()
    assert model.decoder.head[-1].weight.grad[:2].abs().sum()>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_head_initialization_preserves_cuda_rng(frame,cfg):
    ds=CandleWindows(frame,0,600,cfg,1)
    seed_everything(cfg.seed);seq2seq_model(cfg,ds.scaler,'2A');a=torch.cuda.get_rng_state()
    seed_everything(cfg.seed);seq2seq_model(cfg,ds.scaler,'2B');b=torch.cuda.get_rng_state()
    torch.testing.assert_close(a,b)


def test_calibration_no_optimization(frame,cfg):
    ds=CandleWindows(frame,0,600,cfg,1);model=seq2seq_model(cfg,ds.scaler,'2B')
    before={k:v.clone() for k,v in model.state_dict().items()};cfg.calibration_batches=2
    weights=calibrate(model,loader(ds,cfg),cfg,torch.device('cpu'))
    assert weights['calibration']['samples']==2*cfg.batch_size
    assert weights['calibration']['number_of_batches']==2
    assert weights['calibration']['calibration_seed']==model.initialization_seed
    assert weights['calibration']['batch_origin_indices']==[
        ds.starts[:cfg.batch_size].add(48).tolist(),
        ds.starts[cfg.batch_size:2*cfg.batch_size].add(48).tolist()]
    means=weights['calibration']['means']
    assert weights['return']*means['return']==pytest.approx(.5*means['base'])
    assert weights['trend']*means['trend']==pytest.approx(.25*means['base'])
    for key,value in before.items():torch.testing.assert_close(value,model.state_dict()[key])


def test_pooled_gradient_history():
    model=torch.nn.Linear(2,1,bias=False);collector=GradientHistory()
    for gradient in [torch.tensor([[1.,3.]]),torch.tensor([[5.,7.]])]:
        model.weight.grad=gradient;collector.add(model)
    entry=collector.finish(model)['weight']
    assert entry['gradient']['mean']==4
    assert entry['gradient']['std']==pytest.approx(5**.5)
    assert entry['gradient']['min']==1 and entry['gradient']['max']==7
    assert entry['missing_gradient_batches']==0
    assert entry['gradient']['norm']==pytest.approx(84**.5)
    assert entry['missing_gradient_status']=='none'


@pytest.mark.parametrize('tiny_improvement',[False,True])
def test_stage_patience_and_paired_optimizer_restoration(cfg,tmp_path,monkeypatch,tiny_improvement):
    import training
    cfg.teacher_probabilities=[1.,.5,0.];cfg.stage_max_epochs=[8,8,8]
    cfg.stage_patience=[2,2,2];cfg.zero_stage_min_epochs=5
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([1.]))
            self.register_buffer('feature_mean',torch.zeros(5));self.register_buffer('feature_scale',torch.ones(5))
    model=Tiny();initial_steps=[];after=[]
    def fake_train(model,ld,opt,*args):
        initial_steps.append(float(opt.state.get(model.weight,{}).get('step',0)))
        opt.zero_grad();model.weight.sum().backward();opt.step();after.append(float(model.weight.detach()))
        return dict.fromkeys(['total','base','return','trend'],1.),{},1.,1.
    second=.99999 if tiny_improvement else 2
    scores=iter([1,second,3,1,second,3,1,second,3,4,5])
    def fake_eval(*args):
        score=next(scores)
        return {'losses':dict.fromkeys(['total','base','return','trend'],float(score)),
                'derived_trend':dict.fromkeys(['accuracy','macro_f1','balanced_accuracy'],.5),
                'returns':{'mae':1.},'six_hour_returns':{'mae':1.}},{}
    monkeypatch.setattr(training,'train_epoch',fake_train);monkeypatch.setattr(training,'evaluate',fake_eval)
    monkeypatch.setattr(training,'save_history',lambda *args:None)
    _,dirs=create_run(tmp_path,'2A','test')
    durations,history=fit(model,[],[],cfg,{'return':0,'trend':0},torch.device('cpu'),.3,dirs,'fold_01')
    best_epoch=2 if tiny_improvement else 1
    assert durations==[best_epoch]*3  # BEST epochs, not stops [3,3,5].
    summary=json.loads((dirs['history']/'fold_01'/'fit_summary.json').read_text())
    assert summary['stage_best_epoch']==[best_epoch]*3 and summary['stage_stop_epoch']==[3,3,5]
    assert [r['stage_best_epoch'] for r in history]==[best_epoch]*11
    assert [r['stage_stop_epoch'] for r in history]==[3]*3+[3]*3+[5]*5
    assert initial_steps[3]==best_epoch and initial_steps[6]==2*best_epoch
    assert float(model.weight.detach())==pytest.approx(after[5+best_epoch])
    state=torch.load(dirs['history']/'fold_01'/'best_checkpoint.pt',weights_only=True)
    assert state['epoch']==6+best_epoch and state['epochs_executed']==11 and state['stage_epoch']==best_epoch
    assert next(iter(state['optimizer']['state'].values()))['step']==3*best_epoch
    assert [r['teacher_forcing'] for r in history]==[1]*3+[.5]*3+[0]*5


@pytest.mark.parametrize('mean,scale',[
    ([.001,-.002,.002,.004,5.], [.005,.003,.001,.002,.4]),
    ([0.,0.,0.,0.,0.], [1.,1.,1.,1.,1.]),
    ([.5,-.7,.1618548,.9268379,9.21874], [1.,1.,.02372583,.8634195,.1371985]),
])
def test_constrained_extremes_mean_and_gradients(cfg,mean,scale):
    scaler=Standardizer(torch.tensor(mean,dtype=torch.float64),torch.tensor(scale,dtype=torch.float64))
    model=seq2seq_model(cfg,scaler,'2A')
    zero=torch.zeros(2,12,5,requires_grad=True)
    predicted=model.constrain(zero)
    assert predicted.shape==(2,12,5)
    torch.testing.assert_close(model.physical(predicted),scaler.mean.float().expand_as(predicted),atol=2e-6,rtol=2e-6)
    predicted.sum().backward()
    assert torch.isfinite(zero.grad).all() and (zero.grad>0).all()
    extreme=torch.full((2,12,5),-1e6,requires_grad=True)
    constrained=model.constrain(extreme)
    torch.testing.assert_close(constrained[...,:2],extreme[...,:2],rtol=0,atol=0)
    assert (model.physical(constrained)[...,2:]>=0).all()
    assert (scaler.inverse(constrained.double())[...,2:]>=0).all()
    constrained.sum().backward()
    assert torch.isfinite(extreme.grad).all()


def test_classification_branch_and_ce_gradients(frame,cfg):
    ds=CandleWindows(frame,0,600,cfg,1);batch=next(iter(loader(ds,cfg)))
    model=seq2seq_model(cfg,ds.scaler,'2B');reg_inputs=[];cls_inputs=[]
    hooks=[model.decoder.head.register_forward_pre_hook(lambda m,args:reg_inputs.append(args[0])),
           model.trend_head.register_forward_pre_hook(lambda m,args:cls_inputs.append(args[0]))]
    out=model(batch['context'])
    for hook in hooks:hook.remove()
    assert out['trend_logits'].shape==(cfg.batch_size,12,3)
    assert len(reg_inputs)==len(cls_inputs)==12
    assert all(reg is cls for reg,cls in zip(reg_inputs,cls_inputs))
    assert cls_inputs[0].shape[-1]==cfg.decoder_hidden_size+2*cfg.encoder_hidden_size
    losses(model,out,batch,cfg,{'return':0,'trend':1})['trend'].backward()
    for shared in [model.encoder.lstm,model.decoder.lstm,model.attention.energy]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in shared.parameters())


def test_calibration_and_paired_rng_streams(frame,cfg,monkeypatch):
    ds=CandleWindows(frame,0,600,cfg,1);cfg.calibration_batches=2
    a=seq2seq_model(cfg,ds.scaler,'2A',initialization_seed=77)
    train_a=loader(ds,cfg,True);train_b=loader(ds,cfg,True)
    tf_a=torch.Generator().manual_seed(cfg.seed+cfg.teacher_seed_offset)
    tf_b=torch.Generator().manual_seed(cfg.seed+cfg.teacher_seed_offset)
    shuffle_before=train_b.generator.get_state().clone();tf_before=tf_b.get_state().clone()
    seed_everything(313);cpu_before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    weights=calibrate_temporary(ds.scaler,loader(ds,cfg),cfg,torch.device('cpu'))
    torch.testing.assert_close(cpu_before,torch.get_rng_state())
    for before,after in zip(cuda_before,torch.cuda.get_rng_state_all() if cuda_before else []):
        torch.testing.assert_close(before,after)
    torch.testing.assert_close(shuffle_before,train_b.generator.get_state())
    torch.testing.assert_close(tf_before,tf_b.get_state())
    assert weights['calibration']['calibration_seed']==cfg.seed+cfg.calibration_seed_offset
    torch.rand(29);b=seq2seq_model(cfg,ds.scaler,'2B',initialization_seed=77)
    for name,p in a.named_parameters():torch.testing.assert_close(p,dict(b.named_parameters())[name])
    batch_a=next(iter(train_a));batch_b=next(iter(train_b))
    torch.testing.assert_close(batch_a['origin_index'],batch_b['origin_index'])
    # Observe the actual Bernoulli draws made inside each decoder, not just
    # generator states or hypothetical external random masks.
    draws=[];original_rand=torch.rand
    def observed_rand(*args,**kwargs):
        value=original_rand(*args,**kwargs);draws.append(value<.5);return value
    monkeypatch.setattr(torch,'rand',observed_rand)
    a(batch_a['context'],batch_a['target'],.5,tf_a);masks_a=draws[:];draws.clear()
    b(batch_b['context'],batch_b['target'],.5,tf_b)
    assert len(masks_a)==len(draws)==11
    for first,second in zip(masks_a,draws):torch.testing.assert_close(first,second)
    torch.testing.assert_close(tf_a.get_state(),tf_b.get_state())


def test_final_schedule_uses_five_best_epochs_not_stops():
    best=[[1,2,1],[2,3,2],[3,4,3],[4,5,4],[5,6,5]]
    assert final_stage_durations(best,3,5)==[3,4,3]
    with pytest.raises(ValueError):final_stage_durations(best[:-1],3,5)


def test_explicit_gradient_measurements_and_clipping(frame,cfg,monkeypatch):
    import training
    ds=CandleWindows(frame,0,600,cfg,1);model=seq2seq_model(cfg,ds.scaler,'2B')
    cfg.gradient_clip=1e-5;optimizer=training.make_optimizer(model,cfg)
    original_norm=global_gradient_norm;original_clip=torch.nn.utils.clip_grad_norm_;measured=[];events=[]
    def measure(model):
        value=original_norm(model);measured.append(float(value));events.append('measure');return value
    def clip(*args,**kwargs):
        events.append('clip');original_clip(*args,**kwargs)
        return torch.tensor(999999.)  # This deliberately bogus return must be ignored.
    monkeypatch.setattr(training,'global_gradient_norm',measure)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',clip)
    batch=next(iter(loader(ds,cfg)))
    _,stats,pre,post=train_epoch(model,[batch],optimizer,cfg,{'return':1,'trend':1},torch.device('cpu'),
                               .5,torch.Generator().manual_seed(52),'gradient_test')
    assert events==['measure','clip','measure']
    assert pre==pytest.approx(measured[0]) and post==pytest.approx(measured[1])
    assert pre>cfg.gradient_clip and 0<post<=cfg.gradient_clip*1.01
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(s['gradient']['norm']**2 for s in stats.values())**.5==pytest.approx(pre,rel=1e-5)


def test_missing_gradient_status():
    model=torch.nn.Linear(2,1);model.weight.grad=torch.ones_like(model.weight)
    collector=GradientHistory();collector.add(model);stats=collector.finish(model)
    assert stats['bias']['gradient'] is None and stats['bias']['missing_gradient_status']=='all'


def test_repeat_runs_preserve_existing_files(tmp_path):
    first,dirs=create_run(tmp_path,'2A','test');marker=dirs['history']/'marker.txt';marker.write_text('keep')
    second,dirs2=create_run(tmp_path,'2A','test')
    assert first!=second and marker.read_text()=='keep' and dirs2['history']!=dirs['history']
