import json
import numpy as np
import pytest
import torch

from main import run


@pytest.mark.parametrize('experiment',['2A','2B'])
def test_full_pipeline_all_folds_synthetic(frame,cfg,tmp_path,experiment):
    path=tmp_path/'dataset.csv';frame.to_csv(path,index=False)
    cfg.teacher_probabilities=[1.,0.];cfg.stage_max_epochs=[1,1];cfg.stage_patience=[1,1]
    cfg.zero_stage_min_epochs=1;cfg.train_stride=120;cfg.eval_stride=12;cfg.calibration_batches=1
    dirs=run(experiment,'full',cfg,{'dataset_path':path,'output_root':tmp_path/'outputs'})
    manifest=json.loads((dirs['history']/'run.json').read_text())
    assert manifest['status']=='complete' and manifest['folds_completed']==5
    assert manifest['final_stage_durations']==[1,1]
    for fold in range(1,6):
        root=dirs['history']/f'fold_{fold:02d}'
        assert (root/'best_checkpoint.pt').is_file() and (root/'epochs.csv').is_file()
        records=json.loads((root/'parameter_gradient_history.json').read_text())
        assert len(records)==2 and records[-1]['parameters']
    assert (dirs['history']/'final_fit'/'final_checkpoint.pt').is_file()
    for label in ['last_validation','held_out_test']:
        root=dirs['final reconstruction testing']/label
        assert (root/'last_window_candles.png').is_file() and (root/'period_close_comparison.png').is_file()
        assert (root/'actual_predicted_ohlcv.csv').is_file()
        arrays=np.load(root/'predictions_and_attention.npz')
        assert arrays['physical_predictions'].shape[1:]==(12,5)
        assert arrays['attention'].shape[1:]==(12,48)
        metrics=json.loads((root/'metrics.json').read_text())
        assert ('classification_head' in metrics)==(experiment=='2B')
        assert metrics['derived_trend']['actual_class_counts']==metrics['persistence_trend']['actual_class_counts']
    assert (dirs['matrices']/'held_out_test'/'derived_trend_counts.png').is_file()
    assert (dirs['matrices']/'held_out_test'/'derived_trend_normalized.png').is_file()
    aggregate=json.loads((dirs['history']/'metrics.json').read_text())['walk_forward_aggregate']
    assert aggregate['fold_count']==5
    assert sum(aggregate['derived_trend']['actual_class_counts'])==aggregate['samples']*12
