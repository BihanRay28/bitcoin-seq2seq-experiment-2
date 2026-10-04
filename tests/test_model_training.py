import json
import pytest
import torch

from data import CandleWindows, loader
from model import seq2seq_model
from output import create_run, GradientHistory
from training import calibrate, fit, losses, seed_everything
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
    seed_everything(cfg.seed); b=seq2seq_model(cfg,dataset.scaler,'2B'); rng_b=torch.get_rng_state()
    for name,p in a.named_parameters():torch.testing.assert_close(p,dict(b.named_parameters())[name])
    torch.testing.assert_close(rng_a,rng_b)
    a.eval();b.eval();context=torch.stack([dataset[0]['context']])
    torch.testing.assert_close(a(context)['predictions'],b(context)['predictions'])


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


def test_stage_patience_and_paired_optimizer_restoration(cfg,tmp_path,monkeypatch):
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
    scores=iter([1,2,3,1,2,3,1,2,3,4,5])
    def fake_eval(*args):
        score=next(scores)
        return {'losses':dict.fromkeys(['total','base','return','trend'],float(score)),
                'derived_trend':dict.fromkeys(['accuracy','macro_f1','balanced_accuracy'],.5),
                'returns':{'mae':1.},'six_hour_returns':{'mae':1.}},{}
    monkeypatch.setattr(training,'train_epoch',fake_train);monkeypatch.setattr(training,'evaluate',fake_eval)
    monkeypatch.setattr(training,'save_history',lambda *args:None)
    _,dirs=create_run(tmp_path,'2A','test')
    durations,history=fit(model,[],[],cfg,{'return':0,'trend':0},torch.device('cpu'),.3,dirs,'fold_01')
    assert durations==[3,3,5]
    assert initial_steps[3]==1 and initial_steps[6]==2
    assert float(model.weight.detach())==pytest.approx(after[6])
    state=torch.load(dirs['history']/'fold_01'/'best_checkpoint.pt',weights_only=True)
    assert state['epoch']==7 and state['epochs_executed']==11 and state['stage_epoch']==1
    assert next(iter(state['optimizer']['state'].values()))['step']==3
    assert [r['teacher_forcing'] for r in history]==[1]*3+[.5]*3+[0]*5


def test_repeat_runs_preserve_existing_files(tmp_path):
    first,dirs=create_run(tmp_path,'2A','test');marker=dirs['history']/'marker.txt';marker.write_text('keep')
    second,dirs2=create_run(tmp_path,'2A','test')
    assert first!=second and marker.read_text()=='keep' and dirs2['history']!=dirs['history']
