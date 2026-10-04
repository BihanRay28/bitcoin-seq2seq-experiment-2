import numpy as np
import pytest
import torch

from data import CandleWindows, Standardizer, chronological_split, class_distribution, final_datasets, fold_datasets, load_dataset
from main import HERE, configure_backend, load_configuration, validate_configuration, validate_paths
from output import persistence_forecast
from transforms import (historical_volatility, log_returns, reconstruct_ohlcv, transform_ohlcv,
                        trend_labels, six_hour_normalized_return)


def test_roundtrip_zero_volume_returns_and_gradients(frame):
    raw = torch.tensor(frame.iloc[:60,1:].to_numpy(),dtype=torch.float64,requires_grad=True)
    anchor = 29900.
    features = transform_ohlcv(raw,anchor)
    reconstructed = reconstruct_ohlcv(features,anchor)
    torch.testing.assert_close(reconstructed,raw,rtol=1e-10,atol=1e-8)
    previous = torch.cat((torch.tensor([anchor]),raw[:-1,3]))
    torch.testing.assert_close(log_returns(features),torch.log(raw[:,3]/previous))
    assert features[0,4] == 0
    reconstructed.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_recursive_prediction_anchor():
    features = torch.zeros(1,12,5,dtype=torch.float64)
    features[:,:,1] = .01
    reconstructed = reconstruct_ohlcv(features,100.)
    torch.testing.assert_close(reconstructed[0,:,3],100*torch.exp(torch.arange(1,13,dtype=torch.float64)*.01))
    torch.testing.assert_close(reconstructed[0,1:,0],reconstructed[0,:-1,3])


def test_volatility_origin_and_threshold_boundaries():
    features = torch.zeros(2,48,5,dtype=torch.float64)
    features[:,:,1] = torch.arange(48).double()/1000
    expected = features[:,:,1].std(-1,correction=0)
    torch.testing.assert_close(historical_volatility(features),expected)
    assert trend_labels(torch.tensor([-.51,-.5,0,.5,.51]),.5).tolist() == [0,1,1,1,2]
    returns = torch.ones(2,12,dtype=torch.float64)*.001
    torch.testing.assert_close(six_hour_normalized_return(returns,expected,1e-8),
                               returns.sum(-1)/(expected*np.sqrt(12)+1e-8))


def test_constant_scaler_and_invalid_candles():
    features = torch.zeros(80,5,dtype=torch.float64)
    scaler = Standardizer.fit(features,1e-8)
    assert torch.isfinite(scaler.transform(features)).all()
    raw = torch.tensor([[100.,99.,98.,101.,0.]])
    with pytest.raises(ValueError,match='OHLC'): transform_ohlcv(raw,100.)
    with pytest.raises(ValueError): reconstruct_ohlcv(torch.ones(12,5)*-1,100.)


def test_folds_counts_and_boundaries(frame,cfg):
    dev,folds = chronological_split(len(frame),cfg)
    assert len(folds)==5 and folds[-1].validation_end==dev
    for fold in folds:
        train,valid = fold_datasets(frame,fold,cfg)
        assert train[-1]['target_indices'][-1] < fold.train_end
        assert valid[0]['target_indices'][0] == fold.validation_start+48
        assert valid[-1]['target_indices'][-1] < fold.validation_end
        assert valid.scaler is train.scaler and valid.threshold == train.threshold
        assert len(train) == max(0,1+(fold.train_end-1-48-12)//cfg.train_stride)
        assert len(valid) == max(0,1+(fold.validation_end-fold.validation_start-48-12)//cfg.eval_stride)
        s = int(valid.starts[-1]); raw_features = valid.features[s:s+48]
        torch.testing.assert_close(valid[-1]['sigma'],historical_volatility(raw_features))


def test_future_cannot_change_training_statistics(frame,cfg):
    _,folds = chronological_split(len(frame),cfg); fold=folds[0]
    train,valid=fold_datasets(frame,fold,cfg)
    changed=frame.copy()
    changed.loc[fold.validation_start:,['open','high','low','close']]*=2
    train2,valid2=fold_datasets(changed,fold,cfg)
    torch.testing.assert_close(train.scaler.mean,train2.scaler.mean)
    torch.testing.assert_close(train.scaler.scale,train2.scaler.scale)
    assert train.threshold==train2.threshold
    # Mutating targets after a fixed origin cannot change that origin's sigma.
    index=int(valid[0]['target_indices'][0]); changed=frame.copy()
    changed.loc[index:,['open','high','low','close']]*=2
    _,valid3=fold_datasets(changed,fold,cfg)
    torch.testing.assert_close(valid[0]['sigma'],valid3[0]['sigma'])


def test_configurable_strides(frame,cfg):
    a=CandleWindows(frame,0,600,cfg,1)
    b=CandleWindows(frame,0,600,cfg,7,a.scaler,a.threshold)
    assert b.starts.tolist()==list(range(0,600-1-48-12+1,7))


@pytest.mark.parametrize('kind',['duplicate','gap','negative_volume','missing','bad_high','nonfinite'])
def test_validation_rejects_bad_input(frame,cfg,tmp_path,kind):
    bad=frame.copy()
    if kind=='duplicate':bad.loc[1,'datetime']=bad.loc[0,'datetime']
    if kind=='gap':bad=bad.drop(index=1)
    if kind=='negative_volume':bad.loc[1,'volume']=-1
    if kind=='missing':bad.loc[1,'close']=np.nan
    if kind=='bad_high':bad.loc[1,'high']=1
    if kind=='nonfinite':bad.loc[1,'close']=np.inf
    path=tmp_path/'dataset.csv';bad.to_csv(path,index=False)
    with pytest.raises(ValueError):load_dataset(path,cfg)


@pytest.mark.parametrize('field,value',[('horizon',24),('num_layers',1),('train_stride',0),
                                      ('stage_patience',[3]),('teacher_probabilities',[1,.5,.7,0]),
                                      ('zero_stage_min_epochs',21)])
def test_config_rejects_invalid_contract(cfg,field,value):
    setattr(cfg,field,value)
    with pytest.raises(ValueError):validate_configuration(cfg)


def test_relative_paths_not_cwd(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    _,paths=load_configuration()
    assert paths['dataset_path'].name=='BTC_USDT_30m_Binance_20260913_184821.csv'


@pytest.mark.parametrize('syntax',[
    "dataset_path = 'D:\\!Papers\\FinanceBot\\dataset.csv'\noutput_root = 'D:\\!Papers\\FinanceBot\\outputs'",
    'dataset_path = "D:/!Papers/FinanceBot/dataset.csv"\noutput_root = "D:/!Papers/FinanceBot/outputs"',
])
def test_windows_toml_paths(tmp_path,syntax):
    import tomllib
    from pathlib import PureWindowsPath
    path=tmp_path/'paths.txt';path.write_text(syntax,encoding='utf-8')
    parsed=tomllib.loads(syntax)
    assert PureWindowsPath(parsed['dataset_path'])==PureWindowsPath('D:/!Papers/FinanceBot/dataset.csv')
    _,resolved=load_configuration(HERE/'hyperparameters.txt',path)
    if __import__('os').name=='nt':
        assert PureWindowsPath(resolved['dataset_path'])==PureWindowsPath(parsed['dataset_path'])
    else:
        assert 'dataset.csv' in str(resolved['dataset_path'])


def test_configuration_paths_validation(tmp_path):
    import tomllib
    path=tmp_path/'paths.txt';path.write_text('dataset_path = 123\noutput_root = "outputs"')
    with pytest.raises(ValueError,match='strings'):load_configuration(paths=path)
    path.write_text('dataset_path = ""\noutput_root = "outputs"')
    with pytest.raises(ValueError,match='strings'):load_configuration(paths=path)
    with pytest.raises(tomllib.TOMLDecodeError):tomllib.loads('dataset_path = "D:\\!Papers\\data.csv"')
    with pytest.raises(ValueError,match='existing'):validate_paths({'dataset_path':tmp_path/'missing.csv','output_root':tmp_path/'outputs'})
    dataset=tmp_path/'data.csv';dataset.write_text('placeholder')
    with pytest.raises(ValueError,match='directory'):validate_paths({'dataset_path':dataset,'output_root':dataset/'outputs'})
    validate_paths({'dataset_path':dataset,'output_root':tmp_path/'outputs'})


def test_final_scaler_threshold_and_class_reports_ignore_test(frame,cfg):
    dev,_=chronological_split(len(frame),cfg);train,test=final_datasets(frame,dev,cfg)
    changed=frame.copy()
    changed.loc[dev:,['open','high','low','close']]*=3
    changed.loc[dev:,'volume']*=100
    train2,test2=final_datasets(changed,dev,cfg)
    torch.testing.assert_close(train.scaler.mean,train2.scaler.mean)
    torch.testing.assert_close(train.scaler.scale,train2.scaler.scale)
    assert train.threshold==train2.threshold==test2.threshold and test.scaler is train.scaler
    assert class_distribution(train)==class_distribution(train2)
    for partition in (train,test):
        distribution=class_distribution(partition)
        assert sum(distribution['counts'])==len(partition)*12
        assert sum(distribution['proportions'])==pytest.approx(1.)


def test_persistence_uses_observed_candles_only(frame,cfg):
    ds=CandleWindows(frame,0,600,cfg,1);window=ds[0]
    observed=window['raw_context'].numpy()[None]
    original=persistence_forecast(observed,12)
    window['raw_target'].mul_(20)  # Target mutation cannot enter baseline's API.
    np.testing.assert_array_equal(original,persistence_forecast(observed,12))
    np.testing.assert_array_equal(original[0,:,:4],np.full((12,4),observed[0,-1,3]))
    np.testing.assert_array_equal(original[0,:,4],np.full(12,observed[0,-1,4]))


def test_windows_314_compatibility_is_explicit(cfg,monkeypatch):
    import main
    monkeypatch.setattr(main.platform,'system',lambda:'Windows')
    monkeypatch.setattr(main.sys,'version_info',(3,14,0))
    assert configure_backend(cfg,torch.device('cuda'))['cudnn_enabled'] is False
    cfg.cudnn_policy='enabled'
    assert configure_backend(cfg,torch.device('cuda'))['cudnn_enabled'] is True
