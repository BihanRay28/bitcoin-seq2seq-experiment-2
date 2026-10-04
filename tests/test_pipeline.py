import json
import numpy as np
import pytest
import torch

from main import run


@pytest.mark.parametrize('experiment',['2A','2B'])
def test_full_pipeline_all_folds_synthetic(frame,cfg,tmp_path,experiment,monkeypatch):
    import main
    from data import CandleWindows, chronological_split
    path=tmp_path/'dataset.csv';frame.to_csv(path,index=False)
    cfg.teacher_probabilities=[1.,0.];cfg.stage_max_epochs=[1,1];cfg.stage_patience=[1,1]
    cfg.zero_stage_min_epochs=1;cfg.train_stride=120;cfg.eval_stride=12;cfg.calibration_batches=1
    dev,_=chronological_split(len(frame),cfg);final_complete=False;events=[]
    original_fit=main.fit
    def observed_fit(*args,**kwargs):
        nonlocal final_complete
        result=original_fit(*args,**kwargs)
        if args[8]=='final_fit':final_complete=True;events.append('final_fit_complete')
        return result
    def observed_windows(frame,start,end,*args,**kwargs):
        if start==dev:
            assert final_complete, 'Held-out test windows accessed before final fit completed'
            events.append('test_constructed')
        return CandleWindows(frame,start,end,*args,**kwargs)
    monkeypatch.setattr(main,'fit',observed_fit)
    monkeypatch.setattr(main,'CandleWindows',observed_windows)
    dirs=run(experiment,'full',cfg,{'dataset_path':path,'output_root':tmp_path/'outputs'})
    manifest=json.loads((dirs['history']/'run.json').read_text())
    assert manifest['status']=='complete' and manifest['folds_completed']==5
    assert manifest['final_stage_durations']==[1,1]
    assert manifest['final_schedule_source']=='median CV stage_best_epoch'
    assert events==['final_fit_complete','test_constructed']
    assert set(manifest['persistence_definition'])=={'open','high','low','close','volume','log_returns','trend'}
    for fold in range(1,6):
        root=dirs['history']/f'fold_{fold:02d}'
        assert (root/'best_checkpoint.pt').is_file() and (root/'epochs.csv').is_file()
        records=json.loads((root/'parameter_gradient_history.json').read_text())
        assert len(records)==2 and records[-1]['parameters']
        summary=json.loads((root/'fit_summary.json').read_text())
        assert summary['stage_best_epoch']==summary['stage_stop_epoch']==[1,1]
        data=json.loads((root/'data.json').read_text())
        for key in ['training_classes','validation_classes']:
            assert sum(data[key]['counts'])==data[key]['windows']*12
            assert sum(data[key]['proportions'])==pytest.approx(1.)
        for row in json.loads((root/'epochs.json').read_text()):
            assert np.isfinite(row['pre_clip_global_norm']) and np.isfinite(row['post_clip_global_norm'])
    assert (dirs['history']/'final_fit'/'final_checkpoint.pt').is_file()
    for label in ['last_validation','held_out_test']:
        root=dirs['final reconstruction testing']/label
        assert (root/'last_window_candles.png').is_file() and (root/'period_close_comparison.png').is_file()
        assert (root/'actual_predicted_ohlcv.csv').is_file()
        arrays=np.load(root/'predictions_and_attention.npz')
        assert arrays['physical_predictions'].shape[1:]==(12,5)
        assert arrays['attention'].shape[1:]==(12,48)
        assert (arrays['physical_predictions'][...,2:]>=0).all()
        raw=arrays['raw_predictions']
        assert np.isfinite(raw).all() and (raw[...,:4]>0).all() and (raw[...,4]>=0).all()
        assert (raw[...,1]>=np.maximum(raw[...,0],raw[...,3])).all()
        assert (raw[...,2]<=np.minimum(raw[...,0],raw[...,3])).all()
        if experiment=='2B':assert arrays['trend_logits'].shape[1:]==(12,3)
        metrics=json.loads((root/'metrics.json').read_text())
        assert ('classification_head' in metrics)==(experiment=='2B')
        assert metrics['derived_trend']['actual_class_counts']==metrics['persistence_trend']['actual_class_counts']
        assert sum(metrics['derived_trend']['actual_class_proportions'])==pytest.approx(1.)
        assert all(np.isfinite(value) for value in metrics['losses'].values())
    assert (dirs['matrices']/'held_out_test'/'derived_trend_counts.png').is_file()
    assert (dirs['matrices']/'held_out_test'/'derived_trend_normalized.png').is_file()
    aggregate=json.loads((dirs['history']/'metrics.json').read_text())['walk_forward_aggregate']
    assert aggregate['fold_count']==5
    assert sum(aggregate['derived_trend']['actual_class_counts'])==aggregate['samples']*12
